import argparse
import torch
import cv2
import numpy as np
from net import MADFNet
from util.io import load_ckpt
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType


class InferenceWrapper(torch.nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.model = MADFNet(layer_size=7, args=args).to(device)
        load_ckpt(args.snapshot, [('model', self.model)])

        self._mean = torch.from_numpy(np.array([0.485, 0.456, 0.406],
                                               dtype=np.float32))[None, :, None, None].to(device)
        self._std = torch.from_numpy(np.array([0.229, 0.224, 0.225],
                                              dtype=np.float32))[None, :, None, None].to(device)

    def forward(self, x_in, y_in):
        x_in = torch.permute(x_in.unsqueeze(0), (0, 3, 1, 2))
        y_in = torch.permute(y_in.unsqueeze(0), (0, 3, 1, 2))

        x_in = (x_in.float() / 255. - self._mean) / self._std
        y_out = y_in.float() / 255.

        x_out = self.model(x_in * y_out, y_out)[-1]

        result = y_out * x_in + (1 - y_out) * x_out

        result = torch.clamp(255 * (result * self._std + self._mean) + 0.5, 0, 255).byte()
        result = torch.permute(result, (0, 2, 3, 1))

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--list_file', type=str, default='')
    parser.add_argument('--snapshot', type=str, default='./places2.pth')
    parser.add_argument('--n_refinement_D', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--result_dir', type=str, default='results')
    args = parser.parse_args()

    model_path_onnx = args.snapshot + '.onnx'
    model_q_path_onnx = args.snapshot + '_Q.onnx'

    device = torch.device('cuda')
    model = InferenceWrapper(args, device)
    model.eval()

    in_frame = in_frame_show = cv2.imread("./examples/places2/case2_input.png")
    in_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB)
    in_mask = cv2.imread("./examples/places2/case2_mask.png", cv2.IMREAD_UNCHANGED)

    image = torch.from_numpy(in_frame).to(device)
    mask = torch.from_numpy(in_mask).to(device)

    with torch.no_grad():
        image, mask = image.to(device), mask.to(device)
        output = model(image, mask)
        out_frame_show = cv2.cvtColor(output.detach().cpu().numpy()[0], cv2.COLOR_BGR2RGB)

    torch.onnx.export(model, (image, mask), model_path_onnx,
                      input_names=["frame", "mask"],
                      output_names=["result"])

    quantize_dynamic(model_path_onnx, model_q_path_onnx, weight_type=QuantType.QUInt8)
    model_onnx = rt.InferenceSession(model_q_path_onnx)

    onnx_result = model_onnx.run(None, {'frame': in_frame, 'mask': in_mask})[0][0]
    onnx_result = cv2.cvtColor(onnx_result, cv2.COLOR_BGR2RGB)

    frame_show = np.hstack([in_frame_show, out_frame_show, onnx_result])

