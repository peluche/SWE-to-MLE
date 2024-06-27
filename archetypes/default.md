---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "{{ replace .Name "-" " " | title }}"
images:
- "/posts/.../....png"
tags: []
categories: []
resources:
- name: "foo"
  src: "foo.png"
math:
  enable: true
---

