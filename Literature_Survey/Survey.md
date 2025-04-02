# 📚 Literature Survey: Facial Recognition-Based Gender and Age Classification

This survey covers significant research papers and methods that focus on gender and age prediction from facial images using machine learning and deep learning techniques.

---


##  [1] Rothe et al. (2015) – DEX: Deep EXpectation of Apparent Age

- **Paper**: [ICCV 2015](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- **Summary**:
  - Utilized a pre-trained VGG-16 model and fine-tuned it for age estimation.
  - Treated age prediction as a regression task (expected value of softmax).
  - Trained on large-scale IMDb-WIKI dataset (~500k images).
- **Key Takeaways**:
  - Label smoothing and expectation strategy led to better results.
  - Showed the power of transfer learning from classification to regression.

##  [2] Rothe et al. (2015) – DEX: Deep EXpectation of Apparent Age

- **Paper**: [ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf)
- **Summary**:

- **Key Takeaways**:

  

##  [3] Recent Trends (2020+)

| Paper/Title                                         | Summary |
|----------------------------------------------------|---------|
| "Lightweight CNN for Age/Gender Estimation" (2020) | Focus on mobile deployment and low-resource inference |
| "DeepFace for Age-Gender Classification" (Meta AI) | Combines detection, alignment, and classification in one pipeline |
| "Multitask Learning Approaches"                    | Joint training for age, gender, and emotion increases accuracy |

---

## Insights 

- UTKFace and IMDb-WIKI are great starting points for training and benchmarking.
- Transfer learning using MobileNet or VGGFace improves results.
- Data augmentation and preprocessing (face alignment, histogram equalization) matter a lot.
- Gender classification is easier than age prediction (binary vs regression/multi-class).

## 📝 References

1. Rothe et al., "DEX: Deep EXpectation of apparent age", CVPR 2015.  
2. Rasmus Rothe, "IMDB-WIKI – 500k+ face images", 2015.  
3. Various survey papers on deep learning for facial analysis.
