from .operators import *
import torch, json, pandas
import warnings, lmdb, io, av
import random
import numpy as np
import h5py
from PIL import Image


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class LMDBH5UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,      # LMDB base path
        metadata_path=None,  # Cleaned JSON path
        h5_base_path=None,   # H5 folder path
        repeat=1,
        data_file_keys=("video",),
        height=None, width=None,
        num_frames=81,
        is_test=False       
    ):
        super().__init__()
        self.lmdb_base_path = base_path
        self.h5_base_path = h5_base_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys 
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.is_test = is_test

        with open(metadata_path, "r") as f:
            self.data = json.load(f)

        self.envs = {}
        print(f"LMDBUnifiedDataset initialized with {len(self.data)} verified entries.")

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image, (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return torchvision.transforms.functional.center_crop(image, (target_height, target_width))

    def _load_frames_from_blob(self, byteflow):
        container = av.open(io.BytesIO(byteflow))
        frames = [Image.fromarray(f.to_ndarray(format='rgb24')) for f in container.decode(video=0)]
        if not frames: return [], (0, 0)
        
        orig_w, orig_h = frames[0].size 
        h, w = self.height, self.width
        if h and w:
            frames = [self.crop_and_resize(f, h, w) for f in frames]
            
        return frames, (orig_w, orig_h)

    def _get_item_internal(self, data_id):
        entry = self.data[data_id % len(self.data)].copy()
        vid_id = entry["video"]
        key_raw = entry["key"]
        
        # Format the key string (e.g., 61265 -> "00061265")
        key_str = str(key_raw).zfill(8)

        h5_path = os.path.join(self.h5_base_path, f"{vid_id}.mano.h5")
        try:
            with h5py.File(h5_path, "r") as f:
                if key_str not in f:
                    warnings.warn(f"Key {key_str} not found in {h5_path}")
                    return None
                
                group = f[key_str]
                
                # Retrieve data from the sub-groups
                # Structure assumption: {key}/mano_joints/joints_local
                joints_local = group['mano_joints/joints_local'][:] # (F, 2, 21, 3)
                right = group['mano_joints/right'][:]                    # (F, 2)
                T_global = group['mano_joints/T_global'][:]              # (F, 2, 3)
                K = group['mano_joints/K'][:]                            # (3, 3)
                if joints_local.shape[1] != 2:
                    warnings.warn(f"Data shape error for {vid_id} / {key_str}")
                    return None
                if right.shape[1] != 2:
                    right = right[:, :2]
                if T_global.shape[1] != 2:
                    T_global = T_global[:, :2, :]

                # MANO theta45 for EgoControl-style conditioning (optional key)
                has_theta = 'video/theta45' in group
                if has_theta:
                    theta45_all  = group['video/theta45'][:]  # (F, 3, 45)
                    root_aa_all  = group['video/root_aa'][:]  # (F, 3, 3)
                    right_th_all = group['video/right'][:]    # (F, 3)
                else:
                    theta45_all = root_aa_all = right_th_all = None

        except Exception as e:
            warnings.warn(f"H5 Read Error for {vid_id} / {key_str}: {e}")
            return None

        if vid_id not in self.envs:
            path = os.path.join(self.lmdb_base_path, f"{vid_id}.lmdb")
            if not os.path.exists(path): 
                warnings.warn(f"LMDB Read Error for {vid_id}: {e}")
                return None
            # Standard LMDB open options for speed
            self.envs[vid_id] = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
        
        with self.envs[vid_id].begin(write=False) as txn:
            vid_blob = txn.get(f"{key_str}_video".encode("ascii"))
        
        if not vid_blob: 
            warnings.warn(f"Empty LMDB for {vid_id}: {e}")
            return None

        video_frames, (orig_w, orig_h) = self._load_frames_from_blob(vid_blob)
        total_frames = len(video_frames)

        target_w, target_h = self.width, self.height

        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        offset_x = (new_w - target_w) / 2.0
        offset_y = (new_h - target_h) / 2.0

        K_resized = K.copy()

        K_resized[0, 0] *= scale  # fx
        K_resized[1, 1] *= scale  # fy
        K_resized[0, 2] *= scale  # cx
        K_resized[1, 2] *= scale  # cy

        K_resized[0, 2] -= offset_x
        K_resized[1, 2] -= offset_y
                
        if total_frames < self.num_frames:
            warnings.warn(f"total frames error for {vid_id}: {e}")
            return None
        
        if total_frames != len(joints_local):
            warnings.warn(f"frames mismatch error for {vid_id}: {e}")
            return None

        if self.is_test:
            # Grab exactly the first num_frames
            indices = np.arange(self.num_frames, dtype=int)
        else:
            # Uniformly sample num_frames across the entire video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # Sliced temporal data
        S_joints = joints_local[indices].astype(np.float32)  # (F_s, 2, 21, 3)
        S_right = right[indices].astype(np.float32)         # (F_s, 2)
        S_T = T_global[indices].astype(np.float32)           # (F_s, 2, 3)

        F_s = S_joints.shape[0]

        S_joints_sem = np.full_like(S_joints, np.nan)  # (F_s, 2, 21, 3)
        S_T_sem      = np.full_like(S_T, np.nan)       # (F_s, 2, 3)

        for f in range(F_s):
            for h in range(2):
                r = S_right[f, h]
                if r == 0:
                    # left hand → channel 0
                    S_joints_sem[f, 0] = S_joints[f, h]
                    S_T_sem[f, 0]      = S_T[f, h]
                elif r == 1:
                    # right hand → channel 1
                    S_joints_sem[f, 1] = S_joints[f, h]
                    S_T_sem[f, 1]      = S_T[f, h]
                # r < 0 or invalid → ignored

        S_joints = S_joints_sem
        S_T      = S_T_sem
        
        joints_cam = S_joints + S_T[:, :, None, :]

        joints_cam = np.nan_to_num(joints_cam, nan=0.0)

        joints_cam = joints_cam.reshape(self.num_frames, 42, 3)

        if has_theta:
            theta45_s  = theta45_all[indices]    # (T, N, 45)  N varies per clip
            root_aa_s  = root_aa_all[indices]    # (T, N, 3)
            right_th_s = right_th_all[indices]   # (T, N)
            T_s = len(indices)
            N_ent = theta45_s.shape[1]           # variable number of entities
            mano_theta = np.zeros((T_s, 2, 48), dtype=np.float32)
            for j in range(N_ent):
                left_m  = (right_th_s[:, j] == 0)
                right_m = (right_th_s[:, j] == 1)
                mano_theta[left_m,  0, :45] = theta45_s[left_m,  j]
                mano_theta[left_m,  0, 45:] = root_aa_s[left_m,  j]
                mano_theta[right_m, 1, :45] = theta45_s[right_m, j]
                mano_theta[right_m, 1, 45:] = root_aa_s[right_m, j]

        if "video" in self.data_file_keys:
            entry['video'] = [video_frames[i] for i in indices]
            if "image" in self.data_file_keys:
                entry['image'] = entry['video'][0]

        entry['joints_cam'] = torch.from_numpy(joints_cam).float().unsqueeze(0)  # (1, F_s, 42, 3)
        entry['K'] = torch.from_numpy(K_resized).float().unsqueeze(0)                      # (1, 3, 3)
        if has_theta:
            entry['mano_theta'] = torch.from_numpy(mano_theta).float().unsqueeze(0)  # (1, T, 2, 48)
        entry['prompt'] = entry.get("text", "")

        return entry

    def __getitem__(self, data_id):
        while True:
            item = self._get_item_internal(data_id)
            if item is not None:
                return item
            
            if self.is_test:
                raise ValueError(f"Failed to load valid data for test index {data_id}. Please check your dataset and the warnings above.")
            data_id = random.randint(0, len(self.data) - 1)

    def __len__(self):
        return len(self.data) * self.repeat