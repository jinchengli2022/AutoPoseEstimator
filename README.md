# AutoPoseEstimator
## Language

- [English](#english)
- [中文](#中文)

### 中文
**AutoPoseEstimator** 是一个结合了 **Foundationpose** 和 **Grounded SAM 2** 的全自动目标姿态估计系统。该系统实现了自动读取外部 **mesh** 文件、自动生成 **mask**，并将数据统筹给 **Foundationpose** 进行目标姿态估计的完整流程。

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


---

### English

**AutoPoseEstimator** is a fully automated object pose estimation system that integrates **FoundationPose** and **Grounded SAM 2**. This system automates the entire workflow, including importing external **mesh** files, generating **masks**, and utilizing **FoundationPose** for object pose estimation.  

## Key Features

- **Mesh Import**: Supports importing external 3D models (**mesh**) and processes them automatically.  
- **Automatic Mask Generation**: Uses **Grounded SAM 2** to automatically generate **masks** for input images, eliminating the need for manual annotation.  
- **Pose Estimation**: The generated **masks** are fed into **FoundationPose** for object pose estimation.  
- **Fully Automated Workflow**: From **mesh** import to pose estimation results, the entire process is fully automated, requiring no human intervention.  

## Pipeline

### **Few-shot Mesh Reconstruction**  

Given a static multi-view capture of the object, we begin the **Trajectory Extraction pipeline** by leveraging [TRELLIS](https://github.com/microsoft/TRELLIS), a framework for **Few-shot Reconstruction**, to reconstruct the **object’s shape and texture** from sparse multi-view captures. Following this, we make **manual adjustments** to refine the **scale information**.  

### **Segmentation-based Image Matching**  

To enable accurate segmentation, we generate **42 template images** using **Physically-Based Rendering (PBR)** from the reconstructed mesh. These images serve as references in the **Segmentation-based Image Matching (SIM)** process, where we evaluate segmentation results by computing **semantic matching score, appearance matching score, and geometric matching score**. By integrating these factors, we compute a **comprehensive similarity score**, which we use to accurately extract the **object’s mask** in the initial frame.  

### **Mask Tracking**  

Once we obtain the initial frame mask, we use it as an initial **mask prompt** for the **SAM 2 video predictor**, enabling **consistent and efficient mask propagation** across subsequent frames. Unlike conventional frame-by-frame segmentation, which processes each frame independently, we use the mask from the previous frame as a prompt input for generating the current frame’s mask, forming a **continuous propagation loop**. This iterative process significantly enhances segmentation accuracy and improves processing efficiency.  

### **6D Object Pose Estimation**  

We then provide the **reconstructed mesh**, **tracked mask**, and **original RGB and depth data** as input to **FoundationPose**, a foundation model for **6D object pose estimation and tracking**. By processing these inputs, we can accurately determine the **position and orientation** of the object in 3D space.  

## **References & Acknowledgments**  

- [FoundationPose](https://github.com/username/cnos)  
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)  
- [SAM6D](https://github.com/JiehongLin/SAM-6D/tree/main)  
- [SAM](https://github.com/facebookresearch/sam)  
- [FastSAM](https://github.com/facebookresearch/fastsam)  

