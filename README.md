# ChallengeHC18

[Automated measurement of fetal head circumference](https://hc18.grand-challenge.org/)

## Challenge goals

During pregnancy, ultrasound imaging is used to measure fetal biometrics. One of these measurements is the fetal head circumference (HC).

The HC can be used to estimate the gestational age and monitor growth of the fetus. The HC is measured in a specific cross section of the fetal head, which is called the standard plane.

<img src="https://hc18.grand-challenge.org/media/cache/71/b8/71b84e841fabd10ef8b033131830fcdc@1.5x.png" alt="Illustration" width="600"/>

 This challenge makes it possible to compare developed algorithms for automated measurement of fetal head circumference in 2D ultrasound images.

## Data description

The dataset for this challenge contains a total of 1334 two-dimensional (2D) ultrasound images of the standard plane that can be used to measure the HC.

The data is divided into a training set of 999 images and a test set of 335 images. The size of each 2D ultrasound image is 800 by 540 pixels with a pixel size ranging from 0.052 to 0.326 mm.

The training set also includes an image with the manual annotation of the head circumference for each HC, which was made by a trained sonographer.

## Metric

The results should be submitted as a csv file which contains 6 columns and 336 rows. The first row should be:

**filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad**

<img src="https://hc18.grand-challenge.org/media/HC18/public_html/GrandChallangeValues_90xwKFs.png" alt="Illustration" width="600"/>

## Score updates

| Date  | Model                    | LB score<br>Mean abs difference (mm) ± std | Rank    | Solution                                                 | weight_name                                                |
| ----- | ------------------------ | ------------------------------------------ | ------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| 21/07 | First commit             | x                                          | x       |                                                          | x                                                          |
| 25/07 | UNet<br>Batchnorm        | 5.99 ± 11.26                               | 554/848 | SGD<br>5 epochs<br>no lr scheduling<br>dice loss         | UNet1_loss=0.63_SGD_ep=5_(216, 320)                        |
| 26/07 | UNet<br>Batchnorm        | 4.26 ± 7.58                                | 511/850 | SGD<br>25 epochs<br>multistepLR<br>dice loss             | UNet1_dice=0.4405_SGD_ep=29_(216, 320)_wd=0_dice_loss      |
| 27/07 | UNet<br>Batchnorm        | 3.97 ± 6.97                                | 503/851 | SGD<br>35 epochs<br>multistepLR<br>dice loss<br>data aug | UNet1_dice=0.438_SGD_ep=23_(216, 320)_wd=0_dice_loss       |
| 27/07 | DilatedUNet<br>Batchnorm | 2.65 ± 6.08                                | 402/852 | SGD<br>35 epochs<br>multistepLR<br>dice loss<br>data aug | DilatedUNet_dice=0.477_SGD_ep=28_(216, 320)_wd=0_dice_loss |
