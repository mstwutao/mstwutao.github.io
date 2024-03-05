---
title: "GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty"
authors:
- admin
- Tie Luo
- Donald Wunsch
#author_notes:
#- "Equal contribution"
#- "Equal contribution"
date: "2023-10-08"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "IEEE International Conference on Image Processing (ICIP), 2023"
publication_short: ""

abstract: Adversarial examples (AE) with good transferability enable practical black-box attacks on diverse target models, where insider knowledge about the target models is not required. Previous methods often generate AE with no or very limited transferability; that is, they easily overfit to the particular architecture and feature representation of the source, white-box model and the generated AE barely work for target, black-box models. In this paper, we propose a novel approach to enhance AE transferability using Gradient Norm Penalty (GNP). It drives the loss function optimization procedure to converge to a flat region of local optima in the loss landscape. By attacking 11 state-of-the-art (SOTA) deep learning models and 6 advanced defense methods, we empirically show that GNP is very effective in generating AE with high transferability. We also demonstrate that it is very flexible in that it can be easily integrated with other gradient based methods for stronger transfer-based attacks.

# Summary. An optional shortened abstract.
summary: IEEE International Conference on Image Processing (ICIP), 2023

#tags:
#- Source Themes
featured: true

# links:
# - name: ""
#   url: ""
url_pdf: https://arxiv.org/pdf/2307.04099.pdf
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ''
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
