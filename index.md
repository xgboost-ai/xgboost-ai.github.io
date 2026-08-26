---
layout: default
title: XGBoost
description: Scalable and flexible gradient boosting
---

<section class="home-hero" aria-labelledby="home-hero-title">
  <div class="home-section-inner home-hero__inner">
    <h1 id="home-hero-title">Scalable and Flexible Gradient Boosting</h1>
    <div class="home-hero__social" aria-label="XGBoost GitHub activity">
      <iframe
        src="https://ghbtns.com/github-btn.html?user=dmlc&amp;repo=xgboost&amp;type=star&amp;count=true&amp;v=2"
        title="Star XGBoost on GitHub"
        width="120"
        height="20"
      ></iframe>
      <iframe
        src="https://ghbtns.com/github-btn.html?user=dmlc&amp;repo=xgboost&amp;type=fork&amp;count=true&amp;v=2"
        title="Fork XGBoost on GitHub"
        width="100"
        height="20"
      ></iframe>
    </div>
    <a class="home-hero__button" href="{{ '/about' | relative_url }}">Get Started</a>
  </div>
</section>

<section class="home-latest" aria-labelledby="home-latest-title">
  <div class="home-section-inner">
    <h2 id="home-latest-title">Latest from the XGBoost Blog</h2>
    <ul class="post-list home-post-list">
      {% for post in site.posts limit:3 %}
      <li>
        <h3>
          <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
        </h3>
        <time class="post-meta" datetime="{{ post.date | date_to_xmlschema }}">
          {{ post.date | date: site.minima.date_format }}
        </time>
      </li>
      {% endfor %}
    </ul>
  </div>
</section>

<section class="home-features" aria-label="XGBoost capabilities">
  <div class="home-section-inner home-feature-grid">
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-flag home-feature__icon" aria-hidden="true"></i>
        Flexible
      </h2>
      <p>Supports regression, classification, ranking, and user-defined objectives.</p>
    </article>
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-cube home-feature__icon" aria-hidden="true"></i>
        Portable
      </h2>
      <p>Runs on Windows, Linux, and macOS, as well as major cloud platforms.</p>
    </article>
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-wrench home-feature__icon" aria-hidden="true"></i>
        Multiple Languages
      </h2>
      <p>Supports multiple languages including C++, Python, R, Java, Scala, and Julia.</p>
    </article>
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-cogs home-feature__icon" aria-hidden="true"></i>
        Battle-tested
      </h2>
      <p>Wins many data science and machine learning challenges and is used in production by multiple companies.</p>
    </article>
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-cloud home-feature__icon" aria-hidden="true"></i>
        Distributed on Cloud
      </h2>
      <p>Supports distributed training across multiple machines and integrates with systems including Spark and Flink.</p>
    </article>
    <article class="home-feature">
      <h2>
        <i class="fa-solid fa-rocket home-feature__icon" aria-hidden="true"></i>
        Performance
      </h2>
      <p>Its optimized backend delivers strong performance with limited resources and scales beyond billions of examples.</p>
    </article>
  </div>
</section>
