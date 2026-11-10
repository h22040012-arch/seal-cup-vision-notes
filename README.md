# 皮碗正反识别：ONNX 权重与工程集成

分享高、低压皮碗正反识别的实际 ONNX 权重、CPU 推理代码、Windows C++ 原生推理与 VisionMaster C# 接入脚本。内容覆盖 ROI、预处理一致性、最高分方向判定、异常处理及宿主集成。

已去除训练元数据中的用户名和本地训练路径。模型图与权重数值保持不变。第三方作者与许可证署名保留，不包含项目使用方的公司、客户或人员名称。

## 直接运行

Python 3.10–3.12：

```bash
pip install -r requirements.txt
python inference.py --station high --image samples/high/front.jpg
python inference.py --station low --image samples/low/front.jpg
python -m unittest discover -s tests -v
python tools/check_models.py
```

两份权重直接存放在 models/，无需另行下载。samples/ 包含真实工件图片，供最小回放使用；部分图片来自既有训练或验证样本，不能用于估计总体准确率。check_models.py 验证摘要并运行六张图，不自动调整门槛。

## 模型与决策

| 模型 | 输入 | 预处理 | 输出 |
|---|---|---|---|
| 高压 high.onnx | [1,3,640,640] | 从 4024×3036 原图外部裁剪 ROI，再 letterbox / RGB / NCHW / ÷255 | [1,300,6] |
| 低压 low.onnx | [1,3,3036,4024] | 外部只做 RGB / NCHW / 归一化，ROI 和缩放在图内 | [1,300,6] |

每行输出为 x1,y1,x2,y2,score,class，类别 0 正面、1 反面。取所有输出行中最高分的一行，同分保留第一行；分数 ≥0.30 时按方向输出正面 OK、反面 NG，否则未检出 NG。没有反面优先、多框否决、面积否决或额外坐标否决。异常输出 NG，不能沿用上一帧 OK。

profiles.ini 保留实际 ROI 与线程配置。兼容字段 OK、MinArea 不参与最终最高分决策；实际使用 Candidate 作为门槛。0.30 是此版本工程配置，不是已证明适用于所有场景的最佳值。

## 代码结构

- inference.py：新增的可独立运行 Python 入口，复用实际模型契约与最终方向策略。
- native/cup_native_top1.cpp：实际 Windows C API 推理源码，包含高压插值、低压全图输入、哈希校验、线程隔离与 ABI。
- integration/VM_High.cs、VM_Low.cs：实际宿主脚本，发布 OK/NG、状态、分数和请求 ID。
- src/seal_vision/：教学策略示例；其更严格的坏框检查不代表原生版本最终规则。
- [工程笔记](docs/engineering.md)：编码、帧身份、验证与回退经验。
- [模型说明](models/MODEL_CARD.md)：来源、许可证、元数据清理与限制。

## Windows 原生集成

需要 Windows x64、C++17 编译器、CMake 及 ONNX Runtime 1.20.1 的 C API 头文件：

```powershell
cmake -S native -B build -A x64 -DONNXRUNTIME_INCLUDE_DIR=C:/onnxruntime/include
cmake --build build --config Release
```

独立部署目录需要 profiles.ini、models/ 和 bin/。将构建的 CupVmInference20261030.dll 放进 bin/；将官方 ONNX Runtime 1.20.1 x64 的 onnxruntime.dll 复制为 bin/CupOrt1201.dll，配套 DLL 也放入该目录。按实际需要安装官方 VC++ x64 运行库。

导入 C# 脚本前设置 NativeRoot，按宿主 SDK 配置图像输入 in0、计数输入 num 和脚本输出。宿主版本及流程由使用者自行适配。仓库不包含厂商 SDK、完整设备工程或第三方运行库二进制，由各自官方渠道提供。CMake 入口与 Python 入口是本次分享新增内容，CMake 构建尚未在公开 CI 中执行。

## 验证与许可

本地验证包括公开权重与原权重输出对比、六张真实图片回放及合成测试。GitHub CI 执行合成测试和真实模型回放。小样本通过不等于产线精度或节拍验收。

权重原有元数据标注 Ultralytics AGPL-3.0，随附许可正文。第三方许可继续适用，见 [第三方说明](THIRD_PARTY_NOTICES.md) 与 [公开范围](docs/publication-scope.md)。模型适用于特定视角、尺寸和 ROI，迁移时需重新验证。

本仓库自有源码按 AGPL-3.0 发布，见 LICENSE。第三方内容仍保留原许可与署名。
