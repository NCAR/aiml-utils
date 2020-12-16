# Add a jupyter notebook as a blog

0. Install aiml-utils or if already installed, do a git pull with master before proceeding. 

1. Write your notebook, which we will assume is called test_blog.ipynb, with comments in Markdown. Be sure to check the header number of the most recently published blog, and set yours to be +1 that number (for example, the very first line in the callbacks blog has the title: 3. Callbacks: Utilities for interacting with ML training). If your blog comes next, make sure the first line is: 4. Your blogs title.

2. Add your blog as the last entry to the registry blog/site/_toc.yml. For example: - file: test_blog.ipynb

3. Rebuild the blog website. First change directory to "aiml-utils/blog", then type: jupyter-book build site (you must have juypter-book, as well as ghp-import installed via pip). Once the site is rebuilt, it will supply you with a local address to view the latest (local) build of the website. When you are happy with your entry, commit the blog to your branch of aiml-utils and issue a pull request. 

4. When the pull request is approved and merged, publish the blog by first changing to the aiml-utils/blog directory. Next, execute the following command (which will ask for your github username and password): ghp-import -n -p -f site/_build/html

5. Check the [updated website](https://ncar.github.io/aiml-utils/home.html) for any mistakes or errors.

Please direct any questions or comments to John Schreck (schreck@ucar.edu) or David John Gagne (dgagne@ucar.edu).

# License
This work is licensed under a GNU General Public License v3.0