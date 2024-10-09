# GRPoseNet：Generalizable and Robust 6D Pose Estimation with Sparse Views from RGB Images
（1）**We address the challenging setting of 6D pose estimation for unseen objects using only sparse views from RGB images.**

（2） **We propose an open-world detector** that leverages large models and a matching design based on semantic and geometric information to achieve zero-shot segmentation and matching tasks. **The view selector and refiner** fully utilizes similar view information through multi-scale information and adaptive weights to alleviate the sparse view problem.

（3）**We construct a novel synthetic dataset named RBMOP**, which can be used to evaluate the robustness of the pose estimation model to background and environmental changes.

（4） **Comprehensive experiments demonstrate that the proposed method achieves high performance across various datasets.** Extensive ablation studies confirm the effectiveness of key components of the proposed network.


**DOWNLOAD**

1.We first publish the datasets we use and release our new background complexity dataset RBMOP https://drive.google.com/file/d/1lhKx8gnOLU_LeCAeIU8YCHhj7_Iqpyuo/view?usp=drive_link.
![fig4](https://github.com/user-attachments/assets/a5f85d87-66da-4fdf-a431-7f45fa7b91a5)
We provide some test results on localization and pose estimation on RBMOP.
![RBMOP-test](https://github.com/user-attachments/assets/0b0742d0-58f9-4468-93e3-9b575280b82d)


2. The Linemod dataset and GenMOP dataset we used are also accompanied by relevant links and results.<br>
   Linemod：https://drive.google.com/file/d/1eHG5xWdvNaMPfe_njuDAYWzkWJuJ9PCf/view?usp=drive_link<br>
   GenMOP：https://drive.google.com/open?id=1w4Zwix4Bw_IxXGkntCeNntUOtAfSeBbY&usp=drive_copy<br>
The results highlight the differences and are shown in the figures below:
![fig5](https://github.com/user-attachments/assets/62f32391-0f70-464f-bced-72e5a638cad2)
![fig6](https://github.com/user-attachments/assets/fe268b5d-5ae4-47cc-ae10-2d31089a6ced)





