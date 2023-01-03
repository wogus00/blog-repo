---
title: "블로그 구축기 (1): Hugo + Github Pages"
date: 2023-01-03T14:00:00-05:00
layout: post
categories: Blog
tags: ['hugo', 'github-pages', blog]
draft: true
image: blog-diagram.png`
description: "Hugo와 Github Pages를 이용하여 블로그 구축하기"
---

## 개요
공부를 처음 시작했을때부터 문서화는 뒷전이였다. 이슈가 생겼을땐 이슈를 해결하는데 급급했고 결국 같은 이슈가 생겼을때 해결방법을 까먹어서 다시 구글링하는 과정을 반복했다. 구글링하는 과정에는 항상 개발블로거들이 작성한 포스트들이 도움이 많이되어 나도 블로그를 구축하기로 결심했다. 

개인적으로 블로그를 구축하는데 3가지 요구사항이 있었다:
1. 깃을 이용한 버전 컨트롤 (version control)
2. 다국어 기능 지원
3. 간단하고 빠름

## 정적 vs 동적 사이트
> `정적` 사이트는 서버에 미리 저장된 파일을 그대로 전달하는 사이트다. 반대로 `동적` 사이트는 사용자의 요청(request)에 따라 생성되는 데이터를 전달하는 사이트다. 블로그같은 문서화를 위한 사이트엔 정적 사이트의 기능으로도 충분하기 때문에 정적 사이트 생성기(static website generator)를 이용하여 블로그를 구축하기로 결심했다.

### Hugo
구글링을 한 결과, 크게 3가지의 정적 사이트 생성 플랫폼이 있었다: `Jekyll`, `Hexo`, 그리고 `Hugo`. 

플랫폼 모두 장단점이 있었지만 빠르고 자체적으로 다국어 기능을 지원하는 `Hugo`를 이용하기로 했다.

## 설치
> 모든 설치는 Mac 기준이다. Windows나 Linux 이용자는 [공식웹사이트](https://gohugo.io/installation/)에서 설치방법을 확인하면 된다.

### Hugo 와 Git 설치
`Hugo`와 `Git` 모두 패키지 매니저 홈브루를 통해서 설치할수있다. 홈브루 설치는 홈브루 웹사이트(https://brew.sh/)를 참고하면 된다.
홈브루 설치 이후 터미널에 아래 커맨드를 입력한다:
```
$ brew install hugo
$ brew install git
```

### 리퍼지토리 2개 생성
**2개의 remote 리퍼지로리가 필요하다**: 하나는 컨텐츠소스를 저장할 목적이고 (`source repository`) 다른 하나는 웹사이트를 배포할 목적으로 (`deployment repository`) 생성할거다.

`source repository`는 아무 이름으로 생성해도 괜찮다. 나는 `blog-repo`로 생성했다.
`deployment repository`는 `Github Pages`에서 요구하는 이름으로 생성해야한다: `<깃허브_아이디>.github.io`.

## 디렉토리 설정

### 소스 리퍼지토리
먼저 로컬 머신에 `source repository`를 `clone`해 온다. 원하는 디렉토리에서 아래 커맨드를 입력한다.
```
$ git clone <소스_리퍼지토리_URL>
```

### Hugo 사이트 생성
이제 `source repository` 디렉토리로 이동한 다음 Hugo 프로젝트를 생성해준다. Hugo는 `TOML`, `YAML`, `JSON`파일을 읽을 수 있다. 나는 개인적으로 `YAML`파일이 친숙하여 `YAML`파일로 프로젝트를 생성했다. 잘 모르겠다면 이후 설명할 테마 documentation에서 추천하는 방식으로 만들면 된다.

터미널에 아래 커맨드를 입력한다
```
$ cd <저장소_이름> # 디렉토리 변경
$ hugo new site <프로젝트_이름> -f yaml 
```

### 배포 submodule 설정
이제 Hugo 프로젝트 폴더로 이동한다. 그리고 `deployment repository` submodule로 지정한다. 만약 `public`폴더 가 이미 생성되어 있다면 삭제 후 지정해준다.
```
$ cd <프로젝트_이름> # 디렉토리 변경
$ rm public # public 폴더가 이미 생성되어있는 경우 삭제
$ git submodule add -b master <배포_리퍼지토리_URL> public
```

현재까지 과정을 정리하자면:
1. Hugo와 Git 로컬 머신에 설치
2. Github에서 remote repository 2개 생성
3. 로컬머신으로 `source repository`를 clone하고 안에 Hugo site 생성
4. Hugo site 폴더안에 `deployment repository`를 `public` 폴더안에 submodule로 지정

## 테마
> 티스토리나 워드프레스 플랫폼과 다르게 테마를 지정하지 않으면 이용자가 모든 부분을 디자인해야한다. 모든 부분을 커스텀할 수 있는것이 장점이나, 나는 그럴 지식도 부족하고 포스팅이 주 목적이기 때문에 테마를 불러오기로 결정했다

### 테마 선택
정식 [웹사이트에](https://themes.gohugo.io/) 가면 지원하는 테마를 확인할 수 있다. 이 포스트 및 블로그에선 `hugo-theme-stack`를 이용하고 있다. 자세한 정보는 테마 제작자의 [웹사이트](https://stack.jimmycai.com/)에서 확인할 수 있다.

### 테마 설치
테마를 정했다면 Hugo 프로젝트의 루트 디렉토리로 이동하여 터미널에 아래 커맨드를 입력해준다:
```
$ git submodule add https://github.com/CaiJimmy/hugo-theme-stack/ themes/hugo-theme-stack
``` 
물론 다른 테마를 선택했다면 해당 테마의 가이드를 따르면 된다.
이후 Hugo 프로젝트 폴더에 `config.yaml` 파일에 아래 스크립트를 추가한다:
```
THEME: hugo-theme-stack
```

## 컨텐츠 생성
> 모든 컨텐츠는 Hugo 프로젝트안에 `contents` 폴더에 업로드 해야한다. 

### 포스트 생성
먼저 새로운 포스트를 생성해본다. Hugo 프로젝트 루트 디렉토리에서 아래 커맨드를 입력한다:
```
$ hugo new posts/포스트-이름.md
```

### 로컬서버에서 테스트
이제 배포를 하기전에 로컬서버에서 사이트를 빌드해본다. 루트 디렉토리에서 아래 커맨드를 입력한다:
```
$ hugo server -D
```
`-D`는 draft 모드인 포스트도 확인할 수 있다. `localhost:1313`에서 사이트가 어떻게 빌드되었는지 확인핸다.

## 배포
로컬서버에서 만족했으면 이제 `deployment repository`에 배포만 하면 된다. Hugo 프로젝트 루트 디렉토리로 이동 후 아래 커맨드를 입력한다.
```
$ hugo -t <테마-이름> # config 파일에 입력한 이름을 입력한다
$ cd public
$ git add --all
$ git commit -m '커밋 메세지'
$ git push origin master
```
`hugo -t <테마-이름>` 커맨드를 입력하면 `public`폴더를 알아서 배포용 파일로 수정한다. 이후 `public` 폴더를 `deployment repository`로 푸쉬하는 방식이다.

## 정리
![diagram](/img/2023/build-blog-1/blog-diagram.png)

위 사진은 내가 블로그 환경을 정리해놓은 diagram이다. 다음 포스트땐 `config`파일을 수정해 보겠다.
