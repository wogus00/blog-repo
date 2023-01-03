---
title: "Building My Blog (1): Hugo and Github Pages"
date: 2023-01-01T21:56:59-05:00
layout: post
categories: Blog
tags: ['hugo', 'blog', 'github pages']
toc: true
draft: false
---

In the past, I didn't really consider documenting what I learned. I would always google my issues and only focused on fixing the issue. This became a problem because I would forget how to resolve the issue. This continued to happen, so I needed to document my learnings so I can go back for a reference. 

Also, many other developer's blogs helped me solve some issues, so some day, I hope I can contribute to other people as well. 

I had few features that I wanted for my blog:
1. Version Control
2. Multilingual feature
3. Simple

## Static Website vs Dynamic Website
> `Static` websites are sites with stable content, whereas `dynamic` websites pull contents with user interactions. For the purpose of blogging, I didn't need features of dynamic website, so I decided to search for static website generators. 
### Hugo
With little bit of research, I realized that there are three main static site generating platforms: `Jekyll`, `Hexo`, and `Hugo`.

Among the three, I chose Hugo because it supports multilingual feature and its build time is the fastest among all.

## Installation
>Please note that this post is based on Mac. For Windows and Linux systems, please refer to the official website.
### Hugo and git
You can install hugo and git through homebrew on terminal
```
$ brew install hugo
$ brew install git
```
### Create Two Repositories
**You need two repositories:** one to store all source and one to deploy the website.

For source repository, you can name it anyways. I will create a repository called `blog-repo`. 
For deployment repository, you must create a repository for github pages: `<GITHUB USERNAME>.github.io`.

## Set Directory Structure
### Source repository
On terminal, go to a desired directory and clone source repository: 
```
$ git clone <REPOSITORY-URL>
```
### Create hugo site
Then, change directory and create new hugo site. If you are more familiar with `YAML` file format, add `-f yaml` to the command:
```
$ cd <REPO-NAME> # change directory
$ hugo new site <SITE-NAME> # create hugo site
# hugo new site <SITE-NAME> -f YAML
```
### Set submodule for deployment
Change directory to hugo site folder. Then set deployment repository as submodule:

```
$ cd <SITE-NAME> # change directory
$ rm public # if public folder already exists
$ git submodule add -b master <REPOSITORY-URL> public 
```

## Install Theme
### Find a theme
You can google hugo themes or look up your favorite one from hugo's [official website](https://themes.gohugo.io/).
This blog's theme is in `hugo-theme-stack`. For more information, visit [here](https://stack.jimmycai.com/)
### Install the theme
Run the following command in the root directory of your hugo site:
```
$ git submodule add https://github.com/CaiJimmy/hugo-theme-stack/ themes/hugo-theme-stack
```
### Configuration
Add the following statement in `config.yaml` file:
```
THEME: hugo-theme-stack
```

## Make new post and Test
### Create new post
On terminal, run the following command to create a new post:
```
$ hugo new posts/YOUR-POST-NAME.md
```
### Test on local server
On root directory of the hugo site, run the following command:
```
$ hugo server
```
This command will let you see how the website will look locally. Go to `localhost:1313`. `-D` will show draft posts as well. 

## Deploy
Once you are ready to make your changes public, go to root diretory of hugo site folder and run the following command:
```
$ hugo -t <THEME-NAME>
$ cd public
$ git add --all
$ git commit -m 'MESSAGE'
$ git push origin master
```
This will automatically change your `public` folder ready to be deployed and push all changes to the deployment repository.

## Summary
![diagram](/img/2023/blog-build/blog-diagram.jpg)

Above is a diagram of how I designed my blog. In next post, I will be making changes to `config.yaml` file for blog configurations.