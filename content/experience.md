---
title: 'Experience'
date: 2023-10-24
type: landing

design:
  spacing: '5rem'

# Note: `username` refers to the user's folder name in `content/authors/`

# Page sections
sections:
  - block: resume-experience
    content:
      username: #admin
      education:
        - area: Ph.D. in Computer Science
          institution: Missouri University of Science and Technology
          date_start: 2018-08-01
          date_end: 2024-06-11
          summary: |
            Thesis on _Why LLMs are awesome_. Supervised by [Prof Joe Smith](https://example.com). Presented papers at 5 IEEE conferences with the contributions being published in 2 Springer journals.
          button:
            text: 'Read Thesis'
            url: 'https://example.com'
        - area: B.S. in Engineering Mechanics
          institution: Huazhong University of Science and Technology
          date_start: 2014-08-31
          date_end: 2018-06-31
          summary: |
            GPA: 3.78/4.0
            
            Courses included:
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
      work:
        - position: Director of Cloud Infrastructure
          company_name: GenCoin
          company_url: ''
          company_logo: ''
          date_start: 2021-01-01
          date_end: ''
          summary: |2-
            Responsibilities include:
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
        - position: Backend Software Engineer
          company_name: X
          company_url: ''
          company_logo: ''
          date_start: 2016-01-01
          date_end: 2020-12-31
          summary: |
            Responsibilities include:
            - Migrated infrastructure to a new data center
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
            - lorem ipsum dolor sit amet, consectetur adipiscing elit
    design:
      # Hugo date format
      date_format: 'January 2006'
      # Education or Experience section first?
      is_education_first: false
  # - block: resume-skills
  #   content:
  #     title: Skills & Hobbies
  #     username: admin
  #   design:
  #     show_skill_percentage: false
  - block: resume-awards
    content:
      title: Certificates
      username: admin
  - block: resume-languages
    content:
      title: Languages
      username: admin
---
