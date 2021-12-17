# Reinforcement Learning Project - Actor-Critic learning to fly
This repository is for solving the normal rotor and tilt-rotor enviroment of quadcopter using actor-critic (A2C and A3C) methods and comparing the performance with DQN and DDQN. The study proposes an A2C with 8 tails for actor and a tail for critic to solve the tiltrotor problem. The propsed approach is basically On-policy advantage actor critic with added entropy loss and advantage.

## Enviroment
 Markup : * Observation Space:
  * Vector of 18 for Normal Rotor (3xError in Desired Position, 3xError in Body Rates, 3xError in Desire Velocity, 3x3 flattened Rotation Matrix)
  * Vector of 22 for Tiltrotor (Normal Observation Vector of 18 + 4 Errors in tilt rotor)
*Action Space – Continuous
  * 4 continuous actions for each normal rotor
  * 4 continuous actions for normal and 4 for tiltrotor
  * Fuzzification (0 – 1 for normal rotors, -1 – 1 for tiltrotors with 0.2 step) or vector quantization
  
![image](https://user-images.githubusercontent.com/55484402/146592780-98571b77-018f-465d-8a63-b4d798c72371.png)

![image](https://user-images.githubusercontent.com/55484402/146592754-f888c283-a7f4-4d33-9352-d6225660ac8b.png)

## Reward Function
The reward function is given below where there is a positive reward for staying alive (not crashing) and penalties for errors in action, velocity, position, roll and pitch.

𝑟_𝑡=𝛽−𝛼_𝑎∥𝑎∥_2−∑129_(𝑘∈{𝑝,𝑣,𝜔})▒  𝛼_𝑘 ├ ∥𝑒_𝑘∥┤_2− ∑129_├ 𝑗∈∣𝜙,𝜃}▒  𝛼_𝑗 ├ ∥𝑒_𝑗∥┤_2−∑129_├ 𝑗∈∣𝜙,𝜃}▒  𝛼_𝑗 ├ ∥𝑒_𝑗∥┤_2![image](https://user-images.githubusercontent.com/55484402/146592873-e9510fd0-5e1e-4e0c-bc21-8c98cf6964fa.png)

 Markup : * β ≥ 0 reward for staying alive

* 𝛂_∗ = 𝑤𝑒𝑖𝑔𝑡ℎ𝑠 𝑓𝑜𝑟 𝑣𝑎𝑟𝑖𝑜𝑢𝑠 𝑡𝑒𝑟𝑚𝑠

* ∥𝒂∥_𝟐 = Penalty for wrong actions

* 𝒆_𝒌  = error in position (ep), velocity (ev), and body rates (eω) 

* 𝒆_𝒋  = error in roll (eφ) , pitch (eθ) 
![image](https://user-images.githubusercontent.com/55484402/146592902-62466b46-e175-4e42-adb5-14df58ae4488.png)


## Results
A2C, A3C, DQN and DDQN were compared with the normal rotor quadcopter enviroment where you can turn on each rotor at different level of speeds. A2C and A3C had the same network architecture and DQN and DDQN had the same architecture and parameters. The comparision is made against time.

![image](https://user-images.githubusercontent.com/55484402/146594183-ee80b8cd-d70e-47b8-b5a7-73897e8e3fcf.png)



https://user-images.githubusercontent.com/55484402/146594520-69391b44-2cdc-4fb5-9a71-f12d904723aa.mp4




## Your first website

**GitHub Pages** is a free and easy way to create a website using the code that lives in your GitHub repositories. You can use GitHub Pages to build a portfolio of your work, create a personal website, or share a fun project that you coded with the world. GitHub Pages is automatically enabled in this repository, but when you create new repositories in the future, the steps to launch a GitHub Pages website will be slightly different.

[Learn more about GitHub Pages](https://pages.github.com/)

## Rename this repository to publish your site

We've already set-up a GitHub Pages website for you, based on your personal username. This repository is called `hello-world`, but you'll rename it to: `username.github.io`, to match your website's URL address. If the first part of the repository doesn’t exactly match your username, it won’t work, so make sure to get it right.

Let's get started! To update this repository’s name, click the `Settings` tab on this page. This will take you to your repository’s settings page. 

![repo-settings-image](https://user-images.githubusercontent.com/18093541/63130482-99e6ad80-bf88-11e9-99a1-d3cf1660b47e.png)

Under the **Repository Name** heading, type: `username.github.io`, where username is your username on GitHub. Then click **Rename**—and that’s it. When you’re done, click your repository name or browser’s back button to return to this page.

<img width="1039" alt="rename_screenshot" src="https://user-images.githubusercontent.com/18093541/63129466-956cc580-bf85-11e9-92d8-b028dd483fa5.png">

Once you click **Rename**, your website will automatically be published at: https://your-username.github.io/. The HTML file—called `index.html`—is rendered as the home page and you'll be making changes to this file in the next step.

Congratulations! You just launched your first GitHub Pages website. It's now live to share with the entire world

## Making your first edit

When you make any change to any file in your project, you’re making a **commit**. If you fix a typo, update a filename, or edit your code, you can add it to GitHub as a commit. Your commits represent your project’s entire history—and they’re all saved in your project’s repository.

With each commit, you have the opportunity to write a **commit message**, a short, meaningful comment describing the change you’re making to a file. So you always know exactly what changed, no matter when you return to a commit.

## Practice: Customize your first GitHub website by writing HTML code

Want to edit the site you just published? Let’s practice commits by introducing yourself in your `index.html` file. Don’t worry about getting it right the first time—you can always build on your introduction later.

Let’s start with this template:

```
<p>Hello World! I’m [username]. This is my website!</p>
```

To add your introduction, copy our template and click the edit pencil icon at the top right hand corner of the `index.html` file.

<img width="997" alt="edit-this-file" src="https://user-images.githubusercontent.com/18093541/63131820-0794d880-bf8d-11e9-8b3d-c096355e9389.png">


Delete this placeholder line:

```
<p>Welcome to your first GitHub Pages website!</p>
```

Then, paste the template to line 15 and fill in the blanks.

<img width="1032" alt="edit-githuboctocat-index" src="https://user-images.githubusercontent.com/18093541/63132339-c3a2d300-bf8e-11e9-8222-59c2702f6c42.png">


When you’re done, scroll down to the `Commit changes` section near the bottom of the edit page. Add a short message explaining your change, like "Add my introduction", then click `Commit changes`.


<img width="1030" alt="add-my-username" src="https://user-images.githubusercontent.com/18093541/63131801-efbd5480-bf8c-11e9-9806-89273f027d16.png">

Once you click `Commit changes`, your changes will automatically be published on your GitHub Pages website. Refresh the page to see your new changes live in action.

:tada: You just made your first commit! :tada:

## Extra Credit: Keep on building!

Change the placeholder Octocat gif on your GitHub Pages website by [creating your own personal Octocat emoji](https://myoctocat.com/build-your-octocat/) or [choose a different Octocat gif from our logo library here](https://octodex.github.com/). Add that image to line 12 of your `index.html` file, in place of the `<img src=` link.

Want to add even more code and fun styles to your GitHub Pages website? [Follow these instructions](https://github.com/github/personal-website) to build a fully-fledged static website.

![octocat](./images/create-octocat.png)

## Everything you need to know about GitHub

Getting started is the hardest part. If there’s anything you’d like to know as you get started with GitHub, try searching [GitHub Help](https://help.github.com). Our documentation has tutorials on everything from changing your repository settings to configuring GitHub from your command line.
