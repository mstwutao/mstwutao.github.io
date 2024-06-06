---
title: "GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Tie Luo
  - Donald Wunsch

# Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: "2023-10-08"
doi: ''

# Schedule page publish date (NOT publication's date).
#publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: "IEEE International Conference on Image Processing (ICIP), 2023"
# publication_short: In *SSCI*

abstract: Adversarial examples (AE) with good transferability enable practical black-box attacks on diverse target models, where insider knowledge about the target models is not required. Previous methods often generate AE with no or very limited transferability; that is, they easily overfit to the particular architecture and feature representation of the source, white-box model and the generated AE barely work for target, black-box models. In this paper, we propose a novel approach to enhance AE transferability using Gradient Norm Penalty (GNP). It drives the loss function optimization procedure to converge to a flat region of local optima in the loss landscape. By attacking 11 state-of-the-art (SOTA) deep learning models and 6 advanced defense methods, we empirically show that GNP is very effective in generating AE with high transferability. We also demonstrate that it is very flexible in that it can be easily integrated with other gradient based methods for stronger transfer-based attacks.


# Summary. An optional shortened abstract.
summary: IEEE International Conference on Image Processing (ICIP), 2023

tags:
  - Adversarial Examples

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: https://arxiv.org/pdf/2307.04099.pdf
#url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
#url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
url_poster: ''
url_project: ''
url_slides: ''
#url_source: 'https://github.com/HugoBlox/hugo-blox-builder'
#url_video: 'https://youtube.com'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  #caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
#   - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

<!-- {{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://docs.hugoblox.com/content/writing-markdown-latex/). -->
