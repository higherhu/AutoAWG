import os
import gc
import imageio
import numpy as np
import torch
import torchvision
import cv2
from einops import rearrange
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import math


def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def draw_grid(frame, sub_h, sub_w, texts):
    # 绘制水平线
    for i in range(1, 3):
        cv2.line(frame, (0, i * sub_h), (frame.shape[1], i * sub_h), (0, 255, 0), 2)

    # 绘制垂直线
    for i in range(1, 3):
        cv2.line(frame, (i * sub_w, 0), (i * sub_w, frame.shape[0]), (0, 255, 0), 2)

    # 添加文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 0, 0)
    thickness = 2

    for ii, text in enumerate(texts):
        if text == "empty":
            continue
        start_w = (ii % 3) * sub_w
        start_h = int(ii / 3) * sub_h
        cv2.putText(frame, text, (start_w + 20, start_h + 50), font, font_scale, font_color, thickness, cv2.LINE_AA)

    return frame


def save_control_img(control_img, control_name, save_path, fps=10):
    # control_img in [b f c h w], value in [0, 1]
    control_img_np = control_img.to(torch.float32).detach().cpu().numpy()
    control_img_np = rearrange(control_img_np, 'b f c h w -> b f h w c').squeeze(0)
    control_img_np = (control_img_np*255).astype(np.uint8)
    imageio.mimwrite(save_path.replace(".gen.mp4", f".{control_name}.mp4"), control_img_np, fps=fps)


def save_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale=False,
    n_rows=6,
    fps=12,
    imageio_backend=True,
    color_transfer_post_process=False,
    save_grid=False,
    width_list=None,
    ):
    sub_h = 540
    sub_w = 960
    indexs = [-1, 6, -1, 1, 3, 5, 2, 0, 4]
    texts = [
        "empty",
        "mid_tele",
        "empty",
        "front_left",
        "mid_center",
        "front_right",
        "rear_left",
        "rear_center",
        "rear_right",
    ]

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    outputs_grid = []
    if save_grid:
        assert width_list is not None
        for frame in outputs:
            frame = np.array(frame)
            view_frames = []
            view_start_x = 0
            for width in width_list:
                view_frames.append(cv2.resize(frame[:, view_start_x : view_start_x + width], (sub_w, sub_h)))
                view_start_x += width

            frame_grid = np.zeros((sub_h * 3, sub_w * 3, 3), dtype=np.uint8)
            for ii, index in enumerate(indexs):
                if index == -1:
                    continue
                start_w = (ii % 3) * sub_w
                start_h = int(ii / 3) * sub_h
                frame_grid[start_h : start_h + sub_h, start_w : start_w + sub_w] = view_frames[index]
            frame_grid = draw_grid(frame_grid, sub_h, sub_w, texts)
            outputs_grid.append(Image.fromarray(frame_grid))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
            if len(outputs_grid) > 0:
                imageio.mimsave(path.replace('.mp4', '_grid.mp4'), outputs_grid, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
            if len(outputs_grid) > 0:
                imageio.mimsave(path.replace('.mp4', '_grid.mp4'), outputs_grid, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)
        if len(outputs_grid) > 0:
            outputs_grid[0].save(path.replace('.mp4', '_grid.mp4'), format='GIF',
                                 append_images=outputs_grid, save_all=True, duration=100, loop=0)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start],
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end],
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video

            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start],
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None):
    if isinstance(input_video_path, str):
        cap = cv2.VideoCapture(input_video_path)
        input_video = []

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 1 if fps is None else int(original_fps // fps)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_count += 1

        cap.release()
    else:
        input_video = input_video_path

    input_video = torch.from_numpy(np.array(input_video))[:video_length]
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

    if validation_video_mask is not None:
        validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
        input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)

        input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
        input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
        input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
    else:
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, :] = 255

    return  input_video, input_video_mask, None


def weighted_mse_loss(input, target, weight, reduction='mean'):
    unreduced_mse = F.mse_loss(input, target, reduction='none')
    weighted_mse = unreduced_mse * weight
    if reduction == 'none':
        return weighted_mse
    elif reduction == 'mean':
        return torch.sum(weighted_mse) / torch.sum(weight)
    elif reduction == 'sum':
        return torch.sum(weighted_mse)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


@torch.no_grad()
def cal_hed_online(hed_model, pixel_values, width_list):
    B, T, C, H, W = pixel_values.shape
    assert C == 3
    device = next(hed_model.parameters()).device
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32).to(device)
    # pixel_values must be in 0~255 in float32
    sketch_list = []
    w_cnt = 0
    for w in width_list:
        edges = hed_model(pixel_values[:, :, :, w_cnt : w_cnt + w])
        # pyramid to same resolution
        edges = [F.interpolate(e, (H, w), mode="bilinear") for e in edges]
        edges = torch.stack(edges, dim=2)
        edge = 1.0 / (1.0 + torch.exp(-torch.mean(edges, dim=2)))
        w_cnt += w
        sketch_list.append(edge)
    sketch = torch.concat(sketch_list, dim=-1)
    # thresh = 0.2
    # sketch = ((sketch.clip(thresh, 1) - thresh) / (1 - thresh)).clip(0, 1)
    sketch = sketch.clip(0, 1)
    sketch = rearrange(sketch, "(b f) c h w -> b f c h w", b=B)
    return sketch


@torch.no_grad()
def cal_leres_online(leres_model, pixel_values, width_list):
    B, T, C, H, W = pixel_values.shape
    assert C == 3
    device = next(leres_model.parameters()).device
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32).to(device)
    # pixel_values in 0~255 in float32 rgb then transform it
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    pixel_values = transform(pixel_values)
    max_batch_size = 64
    chunks = torch.split(pixel_values, max_batch_size, dim=0)
    result_chunks = []
    for chunk in chunks:
        result_list = []
        w_cnt = 0
        for w in width_list:
            input_img = chunk[:, :, :, w_cnt : w_cnt + w]
            # resize to fit the model input
            new_h, new_w = float(H), float(w)
            new_h = int(np.round(new_h / 64.0)) * 64
            new_w = int(np.round(new_w / 64.0)) * 64
            input_img = F.interpolate(input_img, (new_h, new_w), mode="bilinear")
            depth = leres_model.depth_model(input_img)
            depth = F.interpolate(depth, (H, w), mode="bilinear")
            # post process
            numbytes=2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8*numbytes))-1
            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > torch.finfo(depth.dtype).eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = torch.zeros(depth.shape)
            # single channel, 16 bit image
            depth_image = out.to(torch.uint16)
            # convert to uint8
            depth_image = torch.abs(depth_image * (255.0/65535.0)).clamp(0, 255).to(torch.uint8)
            # invert image
            depth_image = torch.bitwise_not(depth_image)
            depth_image = (depth_image.to(torch.float32) / 255.)

            w_cnt += w
            result_list.append(depth_image)
        result = torch.concat(result_list, dim=-1)
        result_chunks.append(result)
    result = torch.concat(result_chunks, dim=0)
    result = result.clip(0, 1)
    result = rearrange(result, "(b f) c h w -> b f c h w", b=B)
    return result


@torch.no_grad()
def cal_lineart_standard(lineart_standard_model, pixel_values, width_list):
    B, T, C, H, W = pixel_values.shape
    assert C == 3
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32)
    # pixel_values in 0~255 in float32 rgb
    guassian_sigma=6.0
    intensity_threshold=8
    ksize = math.ceil(6 * guassian_sigma - 1) | 1
    gaussian_blur = transforms.GaussianBlur(kernel_size=ksize, sigma=guassian_sigma)
    result_list = []
    w_cnt = 0
    for w in width_list:
        input_img = pixel_values[:, :, :, w_cnt : w_cnt + w]
        blurred_image = gaussian_blur(input_img)
        intensity, _ = torch.min(blurred_image - input_img, dim=1, keepdim=True)
        intensity = intensity.clip(0, 255)
        intensity /= max(16, torch.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        intensity = intensity.clip(0, 255) / 255.

        w_cnt += w
        result_list.append(intensity)
    result = torch.concat(result_list, dim=-1)
    result = result.clip(0, 1)
    result = rearrange(result, "(b f) c h w -> b f c h w", b=B)
    return result


@torch.no_grad()
def cal_lineart_online(lineart_model, pixel_values, width_list):
    B, T, C, H, W = pixel_values.shape
    assert C == 3
    device = next(lineart_model.parameters()).device
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32).to(device)
    # pixel_values in 0~255 in float32 rgb scale to 0~1
    pixel_values = pixel_values / 255.0
    result_list = []
    w_cnt = 0
    for w in width_list:
        input_img = pixel_values[:, :, :, w_cnt : w_cnt + w]
        line = lineart_model(input_img)
        w_cnt += w
        result_list.append(line)
    result = torch.concat(result_list, dim=-1)
    result = 1.0 - result.clip(0, 1)
    result = rearrange(result, "(b f) c h w -> b f c h w", b=B)
    return result


@torch.no_grad()
def cal_pidi_online(pidi_model, pixel_values, width_list):
    B, T, C, H, W = pixel_values.shape
    max_batch_size = 64
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32)
    # pixel_values in 0~255 in float32 rgb then scale to 0~1
    pixel_values = pixel_values / 255.  # (0, 255) -> (0, 1)

    chunks = torch.split(pixel_values, max_batch_size, dim=0)
    sketch_chunks = []
    for chunk in chunks:
        sketch_list = []
        w_cnt = 0
        for w in width_list:
            sketch = pidi_model(chunk[:, :, :, w_cnt : w_cnt + w])[-1]
            w_cnt += w
            sketch_list.append(sketch)
        sketch = torch.concat(sketch_list, dim=-1)
        sketch_chunks.append(sketch)
    sketch = torch.concat(sketch_chunks, dim=0)
    thresh = 0.2
    sketch = (
        (sketch.clip(thresh, 1) - thresh) / (1 - thresh)
    ).clip(0, 1)
    sketch = rearrange(sketch, "(b f) c h w -> b f c h w", b=B)
    return sketch


@torch.no_grad()
def cal_detect_online(det_model, pixel_values, width_list):
    class_map_sel = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        6: "train",
        7: "truck",
        9: "traffic light",
        11: "stop sign",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
    }
    B, T, C, H, W = pixel_values.shape
    det_mask = torch.zeros((B * T, 1, H, W), dtype=pixel_values.dtype, device=pixel_values.device)
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) h w c"
    ).to(torch.uint8)
    # pixel_values in 0~255 in uint8
    if getattr(det_model, "task", "detect") == "segment":
        conf_thres = 0.3  # use lower conf_thres for segment model
    else:
        conf_thres = 0.4
    w_cnt = 0
    for w in width_list:
        # BGR data in HWC format in a list
        images = pixel_values[:, :, w_cnt : w_cnt + w]
        images = list(images.cpu().numpy()[:, :, :, ::-1])
        results = det_model(images, conf=conf_thres, verbose=False)
        for i, result in enumerate(results):
            masks = result.masks  # mask
            if masks is not None and masks.shape[0] > 0:
                batch_masks = masks.data.unsqueeze(0)
                resized_masks = F.interpolate(batch_masks, size=(H, w), mode='nearest').squeeze(0)
            boxes = result.boxes  # Boxes object for bounding box outputs
            boxes_cls = boxes.cls  # tensor of class indices
            boxes_prob = boxes.conf  # tensor of confidence scores
            boxes_xyxy = boxes.xyxy  # tensor of bounding box coordinates in xyxy format
            for idx, (cls, prob, xyxy) in enumerate(zip(boxes_cls, boxes_prob, boxes_xyxy)):
                if cls.item() in class_map_sel.keys():
                    # print(f"[YOLO] detected {class_map_sel[cls.item()]} with confidence {prob.item()}")
                    if masks is not None:
                        det_mask[i, :, :, w_cnt : w_cnt + w] = \
                            torch.logical_or(det_mask[i, :, :, w_cnt : w_cnt + w],
                                             resized_masks[idx])
                    else:
                        det_mask[i, :, int(xyxy[1]) : int(xyxy[3]),
                                w_cnt+int(xyxy[0]) : w_cnt + int(xyxy[2])] = 1
        w_cnt += w

    det_mask = rearrange(det_mask, "(b f) c h w -> b f c h w", b=B)
    return det_mask


@torch.no_grad()
def dilate_image(image, kernel_size=3, iterations=1):
    """
    使用最大池化实现图像膨胀

    参数:
    - image: 输入图像张量 [B, C, H, W]
    - kernel_size: 膨胀核大小
    - iterations: 膨胀迭代次数

    返回:
    膨胀后的图像张量
    """
    # 确保填充不超过kernel_size的一半
    pad = min(kernel_size - 1, 1)

    # 多次迭代膨胀
    for _ in range(iterations):
        image = F.max_pool2d(
            image,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        )

    return image

@torch.no_grad()
def cal_mask_online(mask_model, pixel_values, width_list, target_classes=("sky")):
    id_to_cls = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }
    cls_to_id = {v: k for k, v in id_to_cls.items()}
    
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    B, T, C, H, W = pixel_values.shape
    assert C == 3
    device = next(mask_model.parameters()).device
    pixel_values = rearrange(
        pixel_values, "b f c h w -> (b f) c h w"
    ).to(torch.float32).to(device)
    pixel_values = pixel_values / 255.0 # pixel_values must be in 0~1 in float32
    
    max_batch_size = 64
    chunks = torch.split(pixel_values, max_batch_size, dim=0)
    mask_chunks = []
    for chunk in chunks:
        out_list = []
        w_cnt = 0
        for w in width_list:
            batch_out = mask_model(transform(chunk[:, :, :, w_cnt : w_cnt + w]))
            w_cnt += w
            out_list.append(batch_out)
        out = torch.concat(out_list, dim=-1)
        mask_arg = out.argmax(1) # (b f) h w

        # 保留目标类别
        target_ids = [cls_to_id[cls] for cls in target_classes if cls in cls_to_id]
        target_ids.append(10)
        mask = torch.zeros_like(mask_arg, dtype=torch.uint8)
        # 将目标类别的位置设置为 1
        for cls_id in target_ids:
            mask[mask_arg == cls_id] = 1
        mask_chunks.append(mask)
    mask = torch.concat(mask_chunks, dim=0)
    mask = rearrange(mask, "(b f) h w -> b f 1 h w", b=B)
    return mask

def load_deeplabv3plus_model(model_path):
    from control_models.deeplabv3plus.network.modeling import deeplabv3plus_resnet101
    # Load model
    model = deeplabv3plus_resnet101(
        num_classes=19, output_stride=16, pretrained_backbone=False
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["model_state"])
    model.eval()

    return model
