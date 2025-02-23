# AutoPoseEstimator

**AutoPoseEstimator** 是一个结合了 **Foundationpose** 和 **Grounded SAM** 的全自动目标姿态估计系统。该系统实现了自动读取外部 **mesh** 文件、自动生成 **mask**，并将数据统筹给 **Foundationpose** 进行目标姿态估计的完整流程。

## 主要功能

- **Mesh 导入**：支持读取外部的 3D 模型（mesh），并进行自动处理。
- **自动生成 Mask**：使用 **Grounded SAM 2** 自动为输入图像生成 **mask**，无需手动标注。
- **姿态估计**：生成的 **mask** 会交给 **Foundationpose** 进行目标的姿态估计。
- **全自动流程**：整个过程从导入 **mesh** 到生成姿态估计结果，全程无需人工干预，完全自动化。

## 系统流程

1. **导入 Mesh**：从外部文件读取 3D 模型（mesh）。此处可用[TRELLIS](https://github.com/microsoft/TRELLIS)生成mesh+手动调整尺度信息
2. **初始帧 Mask**：通过3D 模型（mesh）生成参考掩码，并利用外观、几何等特征生成参考分数，由**FastSAM/SAM**生成初始帧**mask**
3. **生成 Mask**：使用 **Grounded SAM 2** ，通过读取初始帧**mask**，自动生成后续帧 **mask**。
4. **姿态估计**：将生成的 **mask** 输入到 **Foundationpose**，进行目标的姿态估计。
5. **自动化处理**：整个过程自动化，无需用户干预，适用于各种场景。

## 参考&致谢
- [Foundationpose](https://github.com/username/cnos)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [SAM6D](https://github.com/JiehongLin/SAM-6D/tree/main)
- [SAM](https://github.com/facebookresearch/sam)
- [FastSAM](https://github.com/facebookresearch/fastsam)



