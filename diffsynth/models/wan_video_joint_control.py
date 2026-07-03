import torch
import torch.nn as nn
import torch.nn.functional as F


class HandConditioningModule(nn.Module):
    def __init__(
        self,
        num_joints=42,
        embed_dim=64,
        vae_channels=16,
        output_dim=16,
        canvas_height=480,
        canvas_width=832,
        downsample_rate=8,
        heatmap_sigma=2.0
    ):
        super().__init__()
        self.H, self.W = canvas_height, canvas_width
        self.s = downsample_rate
        self.h_grid, self.w_grid = canvas_height // self.s, canvas_width // self.s
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.sigma = heatmap_sigma

        self.joint_id_embed = nn.Embedding(num_joints, embed_dim // 2)
        self.coord_encoder = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )

        in_channels = embed_dim + vae_channels

        self.causal_conv1 = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1)
        )
        self.causal_conv2 = nn.Conv3d(
            embed_dim,
            output_dim,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1)
        )

        self.cross_norm = nn.LayerNorm(output_dim)

    def get_sinusoidal_encoding(self, coords, dim=128):
        """Standard sinusoidal encoding."""
        B, T, N, _ = coords.shape
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=coords.device).float() * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        args = coords.view(-1, 1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(B, T, N, 3, dim).mean(dim=3)

    def generate_heatmaps(self, uv_coords, sigma=None):
        """
        Vectorized Gaussian heatmap generation.
        NOTE: uv_coords are expected to be in Feature Grid Space (already scaled).
        """
        B, T, N, _ = uv_coords.shape
        u, v = uv_coords[..., 0], uv_coords[..., 1]

        y_range = torch.arange(self.h_grid, device=uv_coords.device)
        x_range = torch.arange(self.w_grid, device=uv_coords.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        dist_sq = (grid_x.view(1, 1, 1, self.h_grid, self.w_grid) - u.view(B, T, N, 1, 1))**2 + \
                  (grid_y.view(1, 1, 1, self.h_grid, self.w_grid) - v.view(B, T, N, 1, 1))**2

        if sigma is None:
            heatmaps = torch.exp(-dist_sq / (2 * self.sigma**2))
        else:
            heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
        return heatmaps

    def downsample_wan_style(self, tensor, ft=4):
        """
        Applies WAN-Move temporal logic:
        t=0: Keep exact.
        t>0: Average every 'ft' frames.
        Input: (B, T, ...)
        Output: (B, 1 + (T-1)//ft, ...)
        """
        if tensor.shape[1] == 1:
            return tensor

        t0 = tensor[:, 0:1]
        t_rest = tensor[:, 1:]

        B, T_rest = t_rest.shape[0], t_rest.shape[1]
        T_segments = T_rest // ft

        if T_segments > 0:
            t_rest = t_rest[:, :T_segments*ft]
            shape_rest = list(t_rest.shape)
            new_shape = [B, T_segments, ft] + shape_rest[2:]

            t_averaged = t_rest.view(*new_shape).mean(dim=2)

            return torch.cat([t0, t_averaged], dim=1)
        else:
            return t0

    def project_joints(self, joints_3d, intrinsics):
        """
        Project 3D joints to 2D using Scaled Intrinsics.
        Returns uv in Feature Grid coordinates.
        Supports both fixed K (B, 3, 3) and per-frame K (B, T, 3, 3).
        """
        B, T, N, _ = joints_3d.shape
        joints_flat = joints_3d.view(B * T, N, 3)

        K_scaled = intrinsics.clone()
        K_scaled[..., :2, :] /= self.s

        if K_scaled.dim() == 3:
            K_flat = K_scaled.repeat_interleave(T, dim=0)
        else:
            K_flat = K_scaled.view(B * T, 3, 3)

        projected = torch.bmm(K_flat, joints_flat.transpose(1, 2))

        z_metric = torch.clamp(projected[:, 2:3, :], min=0.2)
        disparity_norm = (1.0 / z_metric) / 5.0
        uv = projected[:, 0:2, :] / z_metric

        return uv.transpose(1, 2).view(B, T, N, 2), \
               disparity_norm.transpose(1, 2).view(B, T, N, 1)

    def get_occlusion_mask(self, uv_coords, disparity, dist_threshold=1.0):
        """
        Calculates occlusion mask.
        Args:
            uv_coords: (B, T, N, 2)
            disparity: (B, T, N, 1)
        """
        diff = uv_coords.unsqueeze(3) - uv_coords.unsqueeze(2)
        dists = torch.norm(diff, dim=-1)

        is_overlapping = dists < dist_threshold

        disp_i = disparity.unsqueeze(3)
        disp_j = disparity.unsqueeze(2)

        i_is_behind_j = disp_i < (disp_j - 1e-3)

        i_is_behind_j = i_is_behind_j.squeeze(-1)

        occluded_by_any = torch.logical_and(is_overlapping, i_is_behind_j).any(dim=3)

        vis_mask = (~occluded_by_any).float().unsqueeze(-1)

        return vis_mask

    def sift_z0_features(self, z0, heatmaps_0, vis_mask_0):
        """
        Zero-out features from occluded joints at t=0.
        """
        eps = 1e-6
        weights = heatmaps_0 / (heatmaps_0.sum(dim=(-2, -1), keepdim=True) + eps)
        pooled_feats = torch.einsum('bnhw, bchw -> bnc', weights, z0)

        sifted_feats = pooled_feats * vis_mask_0
        return sifted_feats

    def propagate_z0_depth_aware(self, heatmaps_latent, pooled_feats, disparity_latent, scale=20.0):
        """
        Args:
            heatmaps_latent: (B, T, N, H, W) [1, 81, 42, 60, 104]
            pooled_feats: (B, N, C)
            disparity_latent: (B, T, N, 1) [1, 81, 42, 1]
        """
        disp_broadcast = disparity_latent.unsqueeze(-1)

        eps = 1e-6

        logits = torch.log(heatmaps_latent + eps) + (disp_broadcast * scale)

        attn_weights = F.softmax(logits, dim=2)

        splatted_normalized = torch.einsum('btnhw, bnc -> btchw', attn_weights, pooled_feats)

        opacity = heatmaps_latent.sum(dim=2, keepdim=True)
        splatted_final = splatted_normalized * opacity

        return splatted_final

    def forward(self, joints, intrinsics, z0):
        B, T, N, _ = joints.shape
        device = joints.device

        uv_coords, disparity = self.project_joints(joints, intrinsics)
        combined_coords = torch.cat([uv_coords, disparity], dim=-1)

        sin_feat = self.get_sinusoidal_encoding(combined_coords, dim=self.embed_dim//2)
        id_feat = self.joint_id_embed(torch.arange(N, device=device))
        id_feat = id_feat.view(1, 1, N, -1).expand(B, T, -1, -1)
        joint_feats = torch.cat([self.coord_encoder(sin_feat), id_feat], dim=-1)

        vis_mask_full = self.get_occlusion_mask(
            uv_coords,
            disparity,
            dist_threshold=1.0
        )

        heatmaps_full_fix = self.generate_heatmaps(uv_coords)

        vis_mask_0 = vis_mask_full[:, 0]
        heatmaps_0 = heatmaps_full_fix[:, 0]

        pooled_feats = self.sift_z0_features(
            z0,
            heatmaps_0,
            vis_mask_0
        )

        struct_map_full = torch.einsum('btnhw, btnd -> btdhw', heatmaps_full_fix, joint_feats)

        heatmaps_latent = self.downsample_wan_style(heatmaps_full_fix, ft=4)
        struct_map_latent = self.downsample_wan_style(struct_map_full, ft=4)
        disparity_latent = self.downsample_wan_style(disparity, ft=4)

        occ_latent = self.propagate_z0_depth_aware(
            heatmaps_latent,
            pooled_feats,
            disparity_latent,
            scale=5.0
        )

        x = torch.cat([struct_map_latent, occ_latent], dim=2)

        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = F.pad(x, (0, 0, 0, 0, 2, 0), mode='replicate')
        x = self.causal_conv1(x)
        x = F.silu(x)

        x = F.pad(x, (0, 0, 0, 0, 2, 0), mode='replicate')
        x = self.causal_conv2(x)
        x = F.silu(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.cross_norm(x)
        out = x.permute(0, 4, 1, 2, 3).contiguous()

        occ_latent = occ_latent.permute(0, 2, 1, 3, 4).contiguous()

        return out, occ_latent
