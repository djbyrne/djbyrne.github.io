---
layout: post
title: The Road To RL Journeyman
tags: [reinforcment-learning, learning, machine-learning]
excerpt_separator: <!--more-->
---

Since leaving college 2 years ago I have been working as an ML engineer in high performing environment with people
much smarter and more capable than myself. Due to this I have had to put in a lot of time ramping up my own core ML and 
software engineering skills in order to keep up. This hasn't left a lot of room for my real passion, Reinforcement Learning. 
Although I still have a lot to learn, even in some of the foundational topics, I want to prioritize some time to focus 
on the area im passionate. RL is what got me into ML in the first place and looking back over the last two years, I 
haven't made as much progress in this area as I would like, even if I have progressed a lot in other areas.

This leads me to the point of this article. I am setting myself a goal moving from an RL novice, to an Expert! 

## The Goal

Alright, expert may be a bit of a stretch, but its nice to have goals. My goal, is to spend the rest of the 2020 focusing
on studying RL and achieving journeyman status. 

> Journeyman: A journeyman is a worker, skilled in a given trade or craft, who has successfully completed an official apprenticeship.

Currently there are 194 days left in 2020, which is roughly 27 weeks. My goal is to speen at least 1 hour a day Mon-Fri
and 6 hours over Sat-Sun studying. This will give me ~11 hours per work to focus on studying, which is ~297 hours in total
before 2021. Now this may sound like a lot of time, but really its not when you consider that it takes around 10,000 hours 
to master a skill. This is merely setting the foundation for my journey in RL. Also, I actually started a lot of this
work a little under a month ago even before I set myself this challenge when I started working on my current RL repo, 
so that adds another 44 hours to my total timeline. 

The repo will all the algorithms I have learned and implemented can be found here https://github.com/djbyrne/core_rl

![Apprentice To Master: http://jasonzavoda-hallofthemountainking.blogspot.com/2018/03/npc-master-journeyman-apprentice.html]({{ "/assets/img/pexels/novice_to_master.jpg" | relative_url}})

## The Plan

Although I have done several RL side projects and covered a lot of the major algorithms at work or at home, I always focused
on implimenting algorithms quickly and put less focus into fully understanding the core principles. To ensure that there 
are no gaps in my knowledge, I plan to start from the beginning with simple tabluar tabular methods and work my way up.

Buts not all. Not only do I want to increase my knowledge of RL, I want to gain practical experience of building RL 
that not high performing and high quality. This means that every RL technique I study I want to also implement and
try and reproduce the results shown in the accompanying paper. In order to ensure a strong understanding of my subject
I have created the following guide for studying the various algorithms and techniques of reinforcement learning. I have 
dubbed this guide, the "Learning Protocol".

### Learning Protocol

The learning protocol is a set of rules and steps that I have set out to ensure the understanding and profficiency of a 
topic. I can only consider a topic "complete" once I have gone through each of these steps. 

- **High Level Understanding:** blog posts, articles, videos, talks, presentations
- **Low Level Understanding:** Papers, Algorithms
- **Code Review:** GitHub Repos, Tutorials
- **Reimplementation:** Recreate model and results
- **Algorithm Explanation:** ReadMe, blog post, presentation/talk

The idea behind these steps is that it goes from a high level base understanding, all the way down to a full low level 
understanding of the full algorithm. In order to fully understand something I believe you need to be able to implement 
it and also be able to explain it simply.

Similarly to the general learning guide, I have a set of steps/requirements for my implementations.

### Implementation Protocol

All model implementations must complete the following requirements before they are considered complete

- **Modular Design:** code is broken up until logical units. Generally no longer than 20 lines
- **Fully Tested:** All core elements must be tested. Core elements are anything that is critical to the models learning
or is planned to be re-used in further work. i.e pretty much all of it. Ideally, all models will be built with TDD 
principles
- **Documented:** I doubt that I will take the time to generate full documentation, but all models will have relevant 
docstrings, type annotations and most importantly, a ReadMe explaining the key points of the model
- **Results & Metrics:** All models will be run with a series of KPI metrics for both model quality and performance. 
The will then be compared to a baseline when applicable.
- **Implemented in Lightning:** Pytorch Lightning is a fantastic library that helps to enforce standards,
reproducability and simplicity when build models.

### Learning Plan
I have previously fallen into the common trap of jumping from one topic to another while studying ML and not really 
finishing a topic or project to completion. To start to research a topic, find some good resources and get to work, but
a few days later I find a "better" resource or I find a "better" topic. To avoid this I have set up a curriculum for
myself that I am going to follow until completion. As well as this I have set out a loose study plan. I say loose 
because the standards I have set are high and I dont really have a huge amount of time, so anything schedules or 
timelines I set here are just best guesses and could quickly change.


#### Schedule

- **Week 1:** Tabular Methods 
- **Week 2-3:** DQN Methods 
- **Week 4:** Policy Gradients 
- **Week 5-6:** Performance Improvements 
- **Week 7-9:** Actor Critic Methods 
- **Week 10-12:** DPG Methods 
- **Week 12-14:** PPO 
- **Week 14-16:** SAC 
- **Week 17-18:** AlphaZero 
- **Week 19-20:** MuZero
- **Week 21-22:** Curiosity Driven Methods
- **Week 23-N:** Time Buffer (Clean Up, Refactoring, Running experiments)

#### Curriculum
<!--more-->
**Reinforcement Learning: An Introduction (Second Edition)**

[Amazon](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249/ref=dp_ob_title_bk) |
[Free](http://incompleteideas.net/book/the-book-2nd.html)

Authours: Richard Sutton and Andrew Barto


**Deep Reinforcement Learning Hands-On (Second Edition)**

[Amazon](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998/ref=pd_sbs_14_1/132-1078346-6339908?_encoding=UTF8&pd_rd_i=1838826998&pd_rd_r=91d3597d-a650-45f2-b165-5af50e7da3c8&pd_rd_w=gkTl9&pd_rd_wg=XQfeK&pf_rd_p=d28ef93e-22cf-4527-b60a-90c984b5663d&pf_rd_r=JRBTA1QQ2FHXKTVFWZP5&psc=1&refRID=JRBTA1QQ2FHXKTVFWZP5)

Author: Maxim Lapan

**Clean Code**

[Amazon](https://www.amazon.co.uk/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882/ref=asc_df_0132350882/?tag=googshopuk-21&linkCode=df0&hvadid=310913487979&hvpos=&hvnetw=g&hvrand=6297754362327610356&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1007880&hvtargid=pla-435472505264&psc=1&th=1&psc=1)

Author: Robert C. Martin

**Berkley Deep RL Bootcamp 2017**

[Link](https://sites.google.com/view/deep-rl-bootcamp/lectures)

Speakers: P. Abieel, Y. Duan, V. Mnih, A. Karpathy, J. Schulman, X. Chen, C. Finn, S. Levine
