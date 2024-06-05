---
# Leave the homepage title empty to use the site title
title: ""
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: "6rem"

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
      text: ""
      # Show a call-to-action button under your biography? (optional)
      button:
        text: Download CV
        url: uploads/resume.pdf
    design:
      css_class: white
      background:
        color: white
        image:
          # Add your image background to `assets/media/`.
          filename:  #mountains.jpg
          filters:
            brightness: 1.0
          size: actual
          position: center
          parallax: false
  # - block: markdown
  #   content:
  #     title: '📚 My Research'
  #     subtitle: ''
  #     text: |-
  #       Use this area to speak to your mission. I'm a research scientist in the Moonshot team at DeepMind. I blog about machine learning, deep learning, and moonshots.

  #       I apply a range of qualitative and quantitative methods to comprehensively investigate the role of science and technology in the economy.
        
  #       Please reach out to collaborate 😃
  #   design:
  #     columns: '1'
  - block: accomplishments
    content:
      # Note: `&shy;` is used to add a 'soft' hyphen in a long heading.
      title: 'Certifications'
      subtitle:
      # Date format: https://wowchemy.com/docs/customization/#date-format
      date_format: Jan 2006
      # Accomplishments.
      #   Add/remove as many `item` blocks below as you like.
      #   `title`, `organization`, and `date_start` are the required parameters.
      #   Leave other parameters empty if not required.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - certificate_url: https://www.coursera.org/account/accomplishments/certificate/V4ULYM7R4RK2
          date_end: ''
          date_start: '2023-08-21'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: AI for Medical Diagnosis
          url: https://www.coursera.org/account/accomplishments/certificate/V4ULYM7R4RK2
        - certificate_url: https://www.coursera.org/account/accomplishments/certificate/DPDC9RH8VRMT
          date_end: ''
          date_start: '2023-07-21'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: AWS Cloud Technical Essentials
          url: https://www.coursera.org/account/accomplishments/certificate/DPDC9RH8VRMT
        - certificate_url: https://www.coursera.org/account/accomplishments/certificate/6WE6SPRZCZTQ
          date_end: ''
          date_start: '2023-07-21'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: Image and Video Processing
          url: https://www.coursera.org/account/accomplishments/certificate/6WE6SPRZCZTQ
        - certificate_url: https://www.coursera.org/account/accomplishments/certificate/NJQ7EW6JE9T7
          date_end: ''
          date_start: '2023-06-21'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: Advanced Computer Vision with TensorFlow
          url: https://www.coursera.org/account/accomplishments/certificate/NJQ7EW6JE9T7
        - certificate_url: https://www.coursera.org/account/accomplishments/certificate/DM2LV2RRZR6L
          date_end: ''
          date_start: '2023-06-21'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: Advanced Learning Algorithms
          url: https://www.coursera.org/account/accomplishments/certificate/DM2LV2RRZR6L
    design:
      columns: '2'
  - block: collection
    id: news
    content:
      title: Recent News
      subtitle: ''
      text: ''
      # Page type to display. E.g. post, talk, publication...
      page_type: post
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: date-title-summary
      # Reduce spacing
      spacing:
        padding: [0, 0, 0, 0]
  - block: collection
    id: papers
    content:
      title: Featured Publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      view: article-grid
      columns: 2
  - block: collection
    content:
      title: Recent Publications
      text: ""
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      view: citation
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      view: article-grid
      columns: 1
  # - block: cta-card
  #   demo: true # Only display this section in the Hugo Blox Builder demo site
  #   content:
  #     title: 👉 Build your own academic website like this
  #     text: |-
  #       This site is generated by Hugo Blox Builder - the FREE, Hugo-based open source website builder trusted by 250,000+ academics like you.

  #       <a class="github-button" href="https://github.com/HugoBlox/hugo-blox-builder" data-color-scheme="no-preference: light; light: light; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star HugoBlox/hugo-blox-builder on GitHub">Star</a>

  #       Easily build anything with blocks - no-code required!
        
  #       From landing pages, second brains, and courses to academic resumés, conferences, and tech blogs.
  #     button:
  #       text: Get Started
  #       url: https://hugoblox.com/templates/
  #   design:
  #     card:
  #       # Card background color (CSS class)
  #       css_class: "bg-primary-700"
  #       css_style: ""
---
