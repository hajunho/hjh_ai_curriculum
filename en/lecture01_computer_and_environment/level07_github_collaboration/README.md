# Lecture 01 · Level 07 — GitHub — Remote Repositories and Collaboration

> Understand remote repositories (clone/push/pull) and the Pull Request collaboration flow, and practice push/pull using a local bare repository as a "fake GitHub" — no internet needed.

**Difficulty** ⭐⭐ / **Prerequisites** level06 / **Estimated time** 45 min

## 1. Why learn this — the business view

The approval cabinet (repository) you built in level06 lives only on your computer. If the laptop dies, the history dies with it, and teammates cannot see your cabinet. What you need is a **remote repository** — a central cabinet everyone can reach. And the world's largest service renting out such central cabinets is **GitHub**.

GitHub is the developer's collaboration venue, their résumé, and the town square where the world's open source gathers. It matters to non-developers too: you can see where and how your company's development output is managed, understand what the dev team means by "the deploy is delayed because the PR is in review," and judge whether an open-source tool you want to adopt is healthy (recent commits, responsive issues). Finish this level and terms like push/pull/PR start to make sense, and the whole picture of remote collaboration comes into focus.

## 2. Understanding through an analogy

**The remote repository is headquarters' central archive; your repository is a branch office's cabinet.**

- **clone**: a newly appointed branch manager **copying the entire central archive** to set up the branch office. The full history is duplicated, so the branch can work independently afterwards.
- **push**: sending documents approved at the branch (commits) **up to the HQ archive** to make them official records.
- **pull**: fetching the latest documents other branches submitted **from HQ** to bring your own branch cabinet up to date.
- **Pull Request (PR)**: a **formal approval request** saying "please merge my proposed changes into the main storyline." Colleagues review the changes; once approved, they are merged into main. The request accumulates discussion comments and change requests, and the whole process is preserved as a record.

The flow in one line: **start with clone → work on a branch → push it up → get reviewed via PR → join main → colleagues update with pull**.

One fun fact: the HQ archive (the remote repository) has no desks for people to work at. A repository that keeps only history, with no working folder, is called a **bare repository** — and GitHub's servers really are this shape. In today's exercise we create a bare repository on your own machine and use it as a "fake GitHub" — even without internet, the mechanics of push/pull are exactly the same.

## 3. Core concepts

### 3-1. The five remote-collaboration commands

```bash
git clone <address>       # start by duplicating the whole remote repository
git remote -v             # check which remote (nicknamed origin) yours is wired to
git push origin main      # send my commits up to the remote's main
git pull origin main      # fetch and merge the remote's new commits
git fetch                 # fetch only, without merging (a preview)
```

`origin` is the nickname automatically attached to the remote you cloned from. Read it as "headquarters."

### 3-2. What GitHub provides

Git is the tool (a program); GitHub is a **service** renting out central repositories for that tool (similar services: GitLab, Bitbucket). On top of repository hosting, GitHub layers collaboration features: PR reviews, issues (a to-do/bug board), wikis, and automation (CI) that runs tests automatically in response to commits and PRs. Companies typically use **private** repositories, while open source shares code with the world via **public** ones.

### 3-3. Conflicts — collaboration's rite of passage

When two branches change **the same part of the same file** differently and both submit, Git does not judge which side is right — it asks a human. That is a conflict. The file shows both versions side by side between `<<<<<<<`, `=======`, and `>>>>>>>` markers; a person edits it to the preferred version (or a third compromise) and commits again, and it is resolved. It looks scary, but it is merely a polite question: "here are both edits side by side — please decide." The tricks that reduce conflicts: **pull often, and push small and often**.

### 3-4. Collaboration etiquette

- The standard is not to push directly to the main branch, but to go through a branch + PR review.
- Split PRs small. Nobody can properly review a 500-line approval request.
- Pull first, push later: fetching the latest and merging on your side before submitting reduces conflicts.

## 4. Hands-on — main.py

```bash
python3 main.py
```

If git is installed, the script builds **one HQ (a bare remote) and two branch offices (developers A and B)** inside a temporary folder and plays the collaboration scenario automatically. (Without git, it falls back to a concept simulation.)

- **[1] Found HQ**: creates a central archive with no work desks using `git init --bare`.
- **[2] Open the branches**: developers A and B each set up shop with `clone`.
- **[3] A pushes**: A creates a menu file, commits, and pushes to HQ.
- **[4] B pulls**: B pulls, and A's file appears in B's office — the core moment of remote collaboration.
- **[5] Back and forth**: B adds content and pushes; A pulls it down, and both sides end up on the same latest state.
- **[6] The PR flow explained**: prints the sequence by which, on real GitHub, this push would continue into PR → review → merge.

The key insight: **there is nothing special about "remote."** Whether the remote address is an internet URL or a folder path on your machine, push/pull behave identically. Watch the code's `run()` function execute git commands in each office folder, and you can see the structure of "who (cwd) exchanges what with which remote (origin)."

## 5. Try it yourself

1. **(Easy)** In output [4], find how B's office folder's file list differs before and after the pull. *(Hint: before the pull it was empty.)*
2. **(Medium)** Compare the scenario with and without an added "A pulls again" step, and think about what goes wrong if A keeps pushing without ever pulling. *(Hint: pushing while ignorant of B's commits gets rejected by the remote — which is why pull comes first.)*
3. **(Challenge)** Create a (free) GitHub account, make a new repository on the web, and edit-and-commit the README directly in the browser. Then fetch it to your machine with `git clone`. *(Hint: the clone address is under the green Code button on the repository page.)*

## 6. Common mistakes

- **Pushing without pulling**: if the remote holds new commits from someone else, your push is rejected. That is not an error but guidance: "fetch and merge first." Run `git pull`, then push again.
- **Pushing a secret key**: a key pushed once lives in the history forever. On a public repository, scavenger bots harvest it within minutes. Prevention is covered in level09.
- **Confusing clone with a zip download**: a zip download is a history-less file copy. To collaborate you must clone — only then do push/pull work.
- **Equating remote with local**: commit but forget to push, and colleagues see nothing. "commit = branch-office approval, push = dispatch to HQ" are separate steps.

## Coming up next

With collaboration in place, it is time to **hand the repetitive manual chores to the machine**. In level08 we learn to automate repetitive tasks — bulk file sorting, renaming, and the like — with shell scripts and Python. This is where the gap opens between "the person who does a 10-minute chore every week" and "the person who automated it once."
