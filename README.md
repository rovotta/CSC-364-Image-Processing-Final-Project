# CSC-364-Image-Processing-Final-Project
Python implementation of the BM3D image denoising algorithm (Dabov et al., 2007). Implements DCT/WHT transforms, block matching, 3D collaborative filtering with hard-thresholding and  Wiener filtering in two separate stages. Built for educational purposes to be presented at the [2026 Davidson College Verna Miller Case Symposium] (VOTTA BM3D 48x36inches academic poster Davidson (1).pdf).  

# Required Packages:

```pip install Pillow```

```pip install numpy```

> *Math and Random are standard python libraries. Pillow is for working with image pixels and numpy is for efficiency purposes.*

# Important Files:

[AWGN.py](AWGN.py) applies additive white gaussian noise to any image

[bm3d_pure.py] (bm3d_pure.py) from-scratch implementation of algorithm (SLOW!!!!)

[bm3d_efficient.py] (bm3d_efficient.py) adds numpy and multi parellel programming for practicality


## Algorithm Overview:

The **Block Matching and 3D filtering** (BM3D) algorithm works in two stages: **first**, it groups mathematically similar image blocks (8x8 pixels default) into 3D groups, applies a separable 3D transform (2D discrete cosine + Walsh-Hadamard), and suppresses noisey pixels through hard thresholding to produce a basic estimate image. **second**, it refines that estimate using Wiener filtering for a more denoised final result. A pure python implementation with explainatory documentation and formulas is written to make every step of the algorithm transparent and approachable for educational purposes. It was built as my final project for my CSC-364 Image Processing class.


## Features:

- This project can be appllied to any gray-scale jpeg image
- User can toggle sigma values, block sizes, hard threshold multipliers/values, and stepping values to optimize denoising results. 
- User can denoise an image dirrectly, or apply ghite gaussian noise at an inputted sigma value
- 


### ✍️ Authors

Mention who you are and link to your GitHub or organization's website.


## 🚀 Usage

*I would love to add a front end to this project in the future, but for now, everything happens in the terminal:*

```py
>>> import mypackage
>>> mypackage.do_stuff()
'Oh yeah!'
```


## 💭 Feedback and Contributing

Add a link to the Discussions tab in your repo and invite users to open issues for bugs/feature requests.

This is also a great place to invite others to contribute in any ways that make sense for your project. Point people to your DEVELOPMENT and/or CONTRIBUTING guides if you have them.


