---
title: "Building Blog (1): Hugo and Github Pages"
date: 2023-01-03
layout: post
categories: Blog
tags: ['hugo', 'github-pages', blog]
draft: false
description: "Building blog with Hugo and Github Pages"
---

## Introduction
Documentation of my learning has never been on my radar. Whenver I faced an issue, I would always focus on fixing it. Eventually, I forgot how I solved that issue, which made me repeat the process of googling for the solution. I recieved lots of help from other's tech blogs, so I decided to create my own blog for others and myself. 

I had three features that I wanted for my blog:
1. Version Control
2. Multilingual feature
3. Simple and fast

## Static vs Dynamic Website
> `Static` websites are sites with stable content, whereas `dynamic` websites pull contents with user requests. For the purpose of blogging, I didn't need features of dynamic website, so I decided to build my blog with static website generators. 

### Hugo
With little bit of research, I realized that there are three main static site generating platforms: `Jekyll`, `Hexo`, and `Hugo`.

The three platforms have pros and cons, but I decided to use `Hugo` because it was fast and supported multilingual feature native. 

## Installation
>Please note that this post is based on Mac. For Windows and Linux systems, please refer to the [official website](https://gohugo.io/installation/).

### Hugo and git
You can install hugo and git through package manger, homebrew. If you don't have homebrew installed, follow instructions on Homebrew's [website](https://brew.sh/). If homebrew is already installed, run the following command on terminal:
```
$ brew install hugo
$ brew install git
```

### Create Two Repositories
**You need two remote repositories:** one to store all source contents (`source repository`) and one to deploy the website (`deployment repositroy`).

For `source repository`, you can name it anyway you prefer. I will create a repository called `blog-repo`. 
For `deployment repository`, you must name it as Github Pages reguires: `<GITHUB USERNAME>.github.io`.

## Set Directory Structure

### Source repository
On terminal, go to a desired directory and clone `source repository`: 
```
$ git clone <REPOSITORY-URL>
```

### Create hugo site
Now, change directory to `source repository` folder and createa Hugo project. Hugo supports `TOML`, `YAML`, and `JSON` files. Personally I am familiar with `YAML` files, so I created my project based on `YAML`. If you are not sure, go to your choice of theme's (explained later in this post) documentation and install their recommended version.

On terminal, run the following command:
```
$ cd <REPO-NAME> # change directory
$ hugo new site <PROJECT-NAME> -f yaml # create hugo site
```

### Set submodule for deployment
Change directory to Hugo project's folder. Then set `deployment repository` as submodule. If `public` folder already exists, remove it.
```
$ cd <SITE-NAME> # change directory
$ rm public # if public folder already exists
$ git submodule add -b master <REPOSITORY-URL> public 
```

To summarize the process until this point:
1. Install Hugo and Git on local machine
2. Create two remote repositories on Github
3. Clone `source repository` to your local machine and create Hugo site.
4. Set `deployment repository` as submodule inside Hugo project folder as `public`

## Install Theme
> Unlike WordPress or other blogging platform, user must customize all features inside the blog unless user sets a theme. This can be a benefit as user can fully customize the website, but for me, I do not have the knowledge to design a website and blogging is the main purpose, so I decided to use pre-built themes.

### Find a theme
Hugo's [official website](https://themes.gohugo.io/) shows all supported themes. In this post and blog, I am using `hugo-theme-stack`. For more information, visit the creator's [website]](https://stack.jimmycai.com/).

### Install the theme
Once you choose your theme, run the following command in the root directory of your hugo site:
```
$ git submodule add https://github.com/CaiJimmy/hugo-theme-stack/ themes/hugo-theme-stack
```
If you choose other theme, follow the installation guide for that theme. Next, open `config.yaml` file inside hugo site folder and add the following script:
```
THEME: hugo-theme-stack
```

## Make New Contents
> All contents must be created under `contents` folder.

### Create new post
On terminal, run the following command to create a new post:
```
$ hugo new posts/YOUR-POST-NAME.md
```

### Test on local server
Before deployment, build and test your website on your local machine. Go to the root directory and run the following command:
```
$ hugo server -D
```
`-D` will show posts in draft mode. Check your website on `localhost:1313` on your browser.

## Deployment
Once you are ready to make your changes public, you need to deploy it through `deployment repository`. Change directory to root directory of Hugo site and run the following command:
```
$ hugo -t <THEME-NAME> # THEME-NAME should be name in config file
$ cd public
$ git add --all
$ git commit -m 'MESSAGE'
$ git push origin master
```
`hugo -t <THEME-NAME>` command will automatically build website inside `public` folder ready for deployment. Then you push `public` folder to `deployment repository`.

## Summary
![diagram](/img/2023/build-blog-1/blog-diagram.png)

Above is a diagram of how I designed my blog. In next post, I will be making changes to `config.yaml` file for blog configurations.