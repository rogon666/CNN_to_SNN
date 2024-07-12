# CNN to SNN conversion for tumor segmentation 

Convolutional neural networks (CNN) are transformed to spiking neural networks (SNN) to classifiy images with and without breast cancer.
SNNs are neural networks that closely mimic biological neural networks. In SNNs, information is encoded in the timing of spikes and data is passed through
the networks in the form of sparse sequences known as Poisson spike trains. Spikes received at a neuron contribute to the membrane potential of the neuron
and the neuron emits a spike or _fires_ when the membrane potential reaches a threshold values. 

The **MatLab** files:

- Train_CNN.m: trains the Convolutional neural networks (CNN) and stores the optimal CNN model as a net object in the file CNN.mat
- Test_CNN.m: loads the optimal CNN model stored in CNN.mat and tests the model in the hold-out sample using accuracy and Qc metrics
- Train_SNN.m: converts the CNN to SNN, trains the SNN model in the train sample, and stores the optimal CNN model as a net object in the file SNN.mat
- Test_SNN.m: loads the optimal SNN model stored in SNN.mat and tests the model in the hold-out sample using accuracy and Qc metrics

## CNN Segmentation Quality Metric: Qc

This repository contains code to evaluate the quality of segmentation performed by a convolutional neural network (CNN) in detecting 'Cancer' regions in images. The `Qc` metric is used to measure the overlap between the predicted 'Cancer' regions and the ground truth 'Cancer' regions.

### Qc Metric Calculation

The `Qc` metric combines precision and recall to provide a balanced measure of segmentation quality. Here's how it's calculated:

1. **Ground Truth (`GT`) Processing**:
   - The ground truth image `GT` is binarized such that all non-zero values are set to 1:
     ```matlab
     GT(GT>0) = 1;
     ```

2. **Segmentation Result (`B`)**:
   - The input image `im` is passed through the semantic segmentation network `net` to get the predicted class labels `C`.
   - A binary mask `B` is created where the predicted class is 'Cancer':
     ```matlab
     [C, scores] = semanticseg(im, net);
     B = (C == 'Cancer');
     ```

3. **Metric Calculation**:
   - `nResult`: The total number of pixels predicted as 'Cancer' by the CNN.
   - `nGT`: The total number of pixels marked as 'Cancer' in the ground truth.
   - `nUNI`: The number of pixels that are predicted as 'Cancer' by the CNN and also marked as 'Cancer' in the ground truth.

4. **Qc Metric**:
   - The `Qc` metric is calculated using the formula:
     ```matlab
     Qc = (nUNI / nGT) * (nUNI / nResult);
     ```
   - Here, `nUNI/nGT` represents the fraction of ground truth 'Cancer' pixels that are correctly identified by the CNN.
   - `nUNI/nResult` represents the fraction of predicted 'Cancer' pixels that are correctly identified compared to the total number of pixels predicted as 'Cancer'.

### Code Example

Here's a detailed explanation of the `Qc` metric calculation in the code:

```matlab
% Calculation of the overlap and quality metric Qc
nResult = sum(sum(B == 1)); % Total predicted 'Cancer' pixels
nGT = sum(sum(GT == 1));    % Total ground truth 'Cancer' pixels
nUNI = 0; % Initialize the count of correctly predicted 'Cancer' pixels

% Calculate the correctly predicted 'Cancer' pixels
for w = 1:numel(GT)
    if B(w) == 1 && GT(w) == 1
        nUNI = nUNI + 1;
    end
end

% Quality metric Qc
Qc = (nUNI / nGT) * (nUNI / nResult);
```
## RIDER breast dataset use in the CNN to SNN conversion

The Reference Image Database to Evaluate Therapy Response (RIDER) is a targeted data collection used to generate an initial consensus on how to harmonize data collection and analysis for quantitative imaging methods applied to measure the response to drug or radiation therapy.  The National Cancer Institute (NCI) has exercised a series of contracts with specific academic sites for collection of repeat "coffee break," longitudinal phantom, and patient data for a range of imaging modalities (currently computed tomography [CT] positron emission tomography [PET] CT, dynamic contrast-enhanced magnetic resonance imaging [DCE MRI], diffusion-weighted [DW] MRI) and organ sites (currently lung, breast, and neuro). 

Ideally a patient’s response to neoadjuvant chemotherapy could be observed noninvasively, in the first 2-3 weeks of treatment using an imaging to provide feedback related to the effectiveness of the chosen chemotherapy regimen. This capability would permit individuation of patient care by supporting the opportunity to tailor chemotherapy to a each patient’s response. Functional diffusion mapping (fDM), now called Parametric Response Mapping (PRM) has been proposed as an MRI imaging biomarker for quantifying early brain tumor response to therapy [1-3]. This approach quantifies local apparent diffusion coefficient (ADC) changes in tumors using a voxel-based analysis implemented by rigid registration of the patient’s head between interval exams. The RIDER Breast MRI data set extended this approach by demonstrating ADC changes in 3 of 5 primary breast cancer patients measured in response to onset of neoadjuvant chemotherapy from interval exams separated by only 8-11 days.

The NCI Cancer Research Data Commons (CRDC) provides access to additional data and a cloud-based data science infrastructure that connects data sets with analytics tools to allow users to share, integrate, analyze, and visualize cancer research data.

See for details the [RIDER Breast MRI Collection](https://www.cancerimagingarchive.net/collection/rider-breast-mri/).
