# ROBIN Watermarking Test Analysis Report

Generated on: 2025-08-19 10:53:19

## Test Overview

- **Total Tests**: 180
- **Images Tested**: 3
- **Attack Types**: 60

## Overall Performance Metrics

- **Average Detection Rate**: 0.605
- **Average Precision**: 0.560
- **Average Recall**: 0.544
- **Average F1 Score**: 0.552
- **Average Attribution Accuracy**: 0.751

## Performance by Attack Category


### CROPPING Attacks

- **Tests**: 27
- **Detection Rate**: 0.467
- **F1 Score**: 0.427
- **Attribution Accuracy**: 0.682

### BLUR Attacks

- **Tests**: 27
- **Detection Rate**: 0.682
- **F1 Score**: 0.619
- **Attribution Accuracy**: 0.791

### COMBINED Attacks

- **Tests**: 9
- **Detection Rate**: 0.555
- **F1 Score**: 0.506
- **Attribution Accuracy**: 0.725

### NOISE Attacks

- **Tests**: 27
- **Detection Rate**: 0.583
- **F1 Score**: 0.530
- **Attribution Accuracy**: 0.741

### SCALING Attacks

- **Tests**: 27
- **Detection Rate**: 0.621
- **F1 Score**: 0.570
- **Attribution Accuracy**: 0.756

### SHARPENING Attacks

- **Tests**: 9
- **Detection Rate**: 0.729
- **F1 Score**: 0.660
- **Attribution Accuracy**: 0.812

### ROTATION Attacks

- **Tests**: 27
- **Detection Rate**: 0.557
- **F1 Score**: 0.507
- **Attribution Accuracy**: 0.727

### JPEG Attacks

- **Tests**: 27
- **Detection Rate**: 0.696
- **F1 Score**: 0.636
- **Attribution Accuracy**: 0.797

## Performance by Attack Intensity


### WEAK Intensity

- **Tests**: 57
- **Detection Rate**: 0.745
- **F1 Score**: 0.678
- **Attribution Accuracy**: 0.821

### MEDIUM Intensity

- **Tests**: 69
- **Detection Rate**: 0.606
- **F1 Score**: 0.554
- **Attribution Accuracy**: 0.751

### STRONG Intensity

- **Tests**: 54
- **Detection Rate**: 0.457
- **F1 Score**: 0.415
- **Attribution Accuracy**: 0.676

## Conclusions

This comprehensive test demonstrates the ROBIN watermarking algorithm's robustness across various attack types and intensities. The results show varying performance depending on the specific attack, with generally higher resistance to blur and JPEG compression compared to geometric transformations like rotation and cropping.
