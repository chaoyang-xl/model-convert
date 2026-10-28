import copy
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
INPUT_MODEL = ROOT / "models/onnx/yolov8s-worldv2_indoor.onnx"
OUTPUT_MODEL = ROOT / "models/onnx/yolov8s-worldv2_indoor_split.onnx"

BOX_TENSOR = "/model.22/Mul_2_output_0"
CLS_TENSOR = "/model.22/Sigmoid_output_0"


def shape_to_list(value_info):
    dims = []

    tensor_type = value_info.type.tensor_type

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    return dims


def find_value_info(graph, name):
    all_infos = (
        list(graph.value_info)
        + list(graph.input)
        + list(graph.output)
    )

    for info in all_infos:
        if info.name == name:
            return info

    return None


def main():

    print("加载：", INPUT_MODEL)

    model = onnx.load(str(INPUT_MODEL))

    # 推断中间tensor shape
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph

    box_info = find_value_info(
        graph,
        BOX_TENSOR
    )

    cls_info = find_value_info(
        graph,
        CLS_TENSOR
    )

    print()
    print("找到中间输出：")

    if box_info is not None:
        print(
            "BOX:",
            BOX_TENSOR,
            shape_to_list(box_info)
        )
    else:
        print("BOX shape 未找到")

    if cls_info is not None:
        print(
            "CLS:",
            CLS_TENSOR,
            shape_to_list(cls_info)
        )
    else:
        print("CLS shape 未找到")

    # 清除原始 output0
    graph.ClearField("output")

    # 优先使用shape inference得到的信息
    if box_info is not None:
        graph.output.append(
            copy.deepcopy(box_info)
        )
    else:
        graph.output.append(
            onnx.helper.make_tensor_value_info(
                BOX_TENSOR,
                onnx.TensorProto.FLOAT,
                [1, 4, 8400],
            )
        )

    if cls_info is not None:
        graph.output.append(
            copy.deepcopy(cls_info)
        )
    else:
        graph.output.append(
            onnx.helper.make_tensor_value_info(
                CLS_TENSOR,
                onnx.TensorProto.FLOAT,
                [1, 12, 8400],
            )
        )

    print()
    print("新的模型输出：")

    for output in graph.output:
        print(
            output.name,
            shape_to_list(output)
        )

    # 检查ONNX合法性
    onnx.checker.check_model(model)

    onnx.save(
        model,
        str(OUTPUT_MODEL)
    )

    print()
    print("保存完成：")
    print(OUTPUT_MODEL.resolve())


if __name__ == "__main__":
    main()
