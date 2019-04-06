<h1 align="center">
  <br>
  <a href="https://colorlib.com/wp/template/colid/"><img src="https://i.imgur.com/09sup5P.png" alt="Colid Template" width=100% height=auto></a>
  <br>
  AcademiesHacks Hackathon Website
  <br>
</h1>

<h4 align="center">A minimal, single page hackathon website built on top of <a href="https://colorlib.com/wp/template/colid/" target="_blank">Colid</a>.</h4>

<p align="center">
  <a href="#design-goals">Design Goals</a> •
  <a href="#process-overview">Process Overview</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

## Design Goals

-   Simple, easy to navigate, information - based
    -   Only necessary resources (images, etc) included
    -   Minimal amount of text
    -   Large, eye - catching buttons with clear purposes
-   Smooth User Experience
    -   CSS minimized
    -   Resource size minimized
    -   Javascript placed at bottom

## Process Overview

-   Day 1:
    -   Finding template
    -   Identified multiple potential templates to uses
    -   Narrowed choices down to three
    -   Chose Colid-Colorlib because of simplicity (single page) and similarity to HackTJ Template
-   Day 2:
    -   Created Wireframe
    -   Altered template to fit basic structure of wireframe
        <br>
           ![Image](images/wireframe.PNG)
-   Day 3:
    -   Added FAQs
    -   Removed overlay from template and added aclrendering [cover image for landing page]
    -   Deleted unused sections and gifs that greatly increased landing time
-   Challenges
    -   Loading time, especially for resources such as images and gifs
        -   Used unusedcss.com to remove boilerplate css that was part of template, but was not used in final website
        -   This introduced some challenges in itself because the unusedcss removed important code, such as the navbar
        -   Overall useful; some small issues
-   Slow Scrolling speed
    -   Somewhat helped by removing resources and optimizing CSS

## Prerequisites

<b>Install Bootstrap using Node Package Manager</b>

<figure class="highlight"><pre><code class="language-bash" data-lang="bash">npm install bootstrap@4.0.0-alpha.6</code></pre></figure>

<p>Bootstrap’s <code class="highlighter-rouge">package.json</code> contains some additional metadata under the following keys:</p>

<ul>
  <li><code class="highlighter-rouge">sass</code> - path to Bootstrap’s main <a href="http://sass-lang.com/">Sass</a> source file</li>
  <li><code class="highlighter-rouge">style</code> - path to Bootstrap’s non-minified CSS that’s been precompiled using the default settings (no customization)</li>
</ul>

*<a href = bootstrapcdn.com> From Bootstrap </a>

<b>Install Colid from Wordpress Command line</b>

<p>To install a new theme using SSH:</p>
<ol>
    <li>Log in to the server via SSH as the user (not root, or you will run into permissions problems).</li>

    //Download the the theme package (usually a zip file) to the themes directory, unzip, and then remove the zip file when done.
    <code>

    <br/> [Haneefkhan@host /home/794013/public_html/wp-content/themes/] : wget http://wordpress.org/extend/themes/download/colid.1.2.1.zip
    <br/> [Haneefkhan@host /home/794013/public_html/wp-content/themes/] : gunzip constructor.1.2.1.zip
    <br/> [HaneefKhan@host /home/794013/public_html/wp-content/themes/] : rm colid.1.2.1.zip
    <br/>

     </code>


</ol>
<p>To activate the new theme:</p>
<ol>
    <li>Log in to the WP admin interface.</li>
    <li>Click on the Appearance button/link on the left side of the interface.</li>
    <li>Your new theme should now appear in the list of theme thumbnails. Click on it, then click &#8220;Activate Theme&#8221; near the top right of the preview window.</li>
    <li>The new theme will now be active for your entire site.</li>
</ol>
<p><strong>NOTE</strong>: You may have to clear your browser cache to see the change, even if there are no caching plugins installed.</p>
*from <a href="liquidweb.com">From Liquid Web</a>

## Credits

This software uses the following open source packages and softwares:

-   [Colid Template](https://colorlib.com/wp/template/colid)
-   [Bootstrap](https://getbootstrap.com/)

## License

MIT
