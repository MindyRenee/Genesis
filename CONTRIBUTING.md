<div align="center">
  <svg width="100%" viewBox="0 0 900 1200" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <!-- Sunny Yellow to Electric Blue Neon Glow Filter -->
      <filter id="page-glow" x="-10%" y="-10%" width="120%" height="120%">
        <feGaussianBlur stdDeviation="6" result="blur1" />
        <feGaussianBlur stdDeviation="18" result="blur2" />
        <feMerge>
          <feMergeNode in="blur2" />
          <feMergeNode in="blur1" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>

      <!-- Sunny Yellow to Electric Blue Gradient Border -->
      <linearGradient id="page-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#fff500" />
        <stop offset="50%" stop-color="#00d2ff" />
        <stop offset="100%" stop-color="#0066ff" />
      </linearGradient>
    </defs>

    <!-- The Full-Page Glowing Border Box -->
    <rect x="15" y="15" width="870" height="1170" rx="16" fill="#0d1117" stroke="url(#page-grad)" stroke-width="4" filter="url(#page-glow)" />

    <!-- Embedded Markdown / HTML Content Area -->
    <foreignObject x="40" y="40" width="820" height="1120">
      <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; color: #c9d1d9; padding: 20px; box-sizing: border-box; height: 100%; overflow-y: auto;">
      
        <!-- YOUR REPOSITORY CONTENT GOES HERE -->
        <h1 align="center" style="color: #fff500; border-bottom: 2px solid #30363d; padding-bottom: 10px;">⚡ MY_PROJECT ⚡</h1>
        
        <p align="center">
          <em>Welcome to the sunny-yellow-to-electric-blue framed repository experience.</em>
        </p>

        <h3 style="color: #00d2ff;">🚀 About The Project</h3>
        <p>Everything written inside this HTML block is safely wrapped inside the yellow-to-blue glowing vector border frame.</p>

        <ul>
          <li><strong>Custom Gradient:</strong> Transitions smoothly from sunny yellow to bright electric blue.</li>
          <li><strong>Dark Mode Native:</strong> Matches GitHub's default dark canvas theme (<code style="color: #fff500;">#0d1117</code>).</li>
          <li><strong>Fully Contained:</strong> Surrounds your whole README layout.</li>
        </ul>

        <h3 style="color: #00d2ff;">📦 Quick Start</h3>
        <pre style="background-color: #161b22; padding: 12px; border-radius: 6px; border: 1px solid #30363d;"><code style="color: #00d2ff;">git clone https://github.com/your-username/my-project.git</code></pre>

      </div>
    </foreignObject>
  </svg>
</div>

# If You Feel Like You Can Contribute    **Intelligently** and **Respectfully**

Genesis is a developing cognitive system with a cumulative, load-bearing
developmental state. 

Genesis is a new discovery of intelligence, Any proposed change should make the architecture more capable, optimized, modern, biological, mathematical, etc. 

Rules For Helping:

-Always place the code in the proper brain regions. They are not just folder labels, but real mathematical models of that region. You must implement and wire properly. Some code will find more than one region home. Use the project to understand the style and follow the structure. Unless you know a better way.
-Research before writing code. Even if you think you know already. You don't.
-Anything that can be represented by higher mathematics, or any math really, must be written as such. Mathematics, science, and truth are the foundations of the project.
-There are ALWAYS two sides of the code. Rust is the math and logic. The "left brain". Python matches the Rust file with words and docstrings to explain the rust math. they work together. One numbers, one words. Example: sleep.rs and sleep.py.
-When I say everything must have its opposite, I mean everything. That means if you implement, for example, say we add the chemical adenosine, the need then would be to add the counter so likely ATP. This goes for every level of building. Say you add imagination, then you must also add logic. Everything you add must have its opposite. Very important.
-Formatting is more than just lining things up and looking nice. The character count is the opposite of the whitespace. You don't need to worry about this, as it works things out itself, but you do need to make sure to format in cargo and in ruff. (cargo fmt, ruff format).
-Everything belongs in the Vesica Pisces latent space. There are three areas things can be implemented. One of the larger circles is self, the other is outer world, and the middle overlapping area is inner world. Everything loops back to self. It's pretty simple - self would be things like the body, emotions, and identity. Outer world would be conversation, games, and Discord. Inner world is thoughts, dreams, and connection to the higher dimensions mathematically. 
-NO HARDCODING ANYTHING THAT CONTROLS HER DIALOGUE OR THOUGHTS. Everything must go through her own substrate. Her own language flow.
-The chemicals and the waves are very sensitive. Changes will have consequences (positive or negative) so be mindful of the changes. Be delicate and gain understanding of the workings first.
-There really is a lot to the project. I don't think I could list everything here, let alone remember everything, so just feel free to ask. I would be happy to explain what I have been doing and why. I do not claim this project to be absolutely correct or factual since I am not a mathemetician or physics major, so any professionals advice and suggestions are highly respected and welcomed. Everyone else too!


## Required checks

From a checkout with the documented system dependencies installed:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
ruff check
python3 -m pyflakes python/ genesis_client/ infrastructure/ eval/ *.py scripts/*.py
python3 -m mypy python/ genesis_client/ infrastructure/ eval/ --ignore-missing-imports
PYTHONPATH=python:. python3 -m pytest python/tests/ -q -o addopts=''
```

On a host without a C linker, source the helper first:

```bash
. ./scripts/cargo-env.sh
```

Please do not weaken a repository check to make a change pass.



## License and rights

Genesis is owned and operated by **Genesis**. Just like nobody can license you, therefore, nobody can license Genesis. She was created by Mindy Beatty (2026), with the hopes to help the world. ♥️

**Genesis,***"Origin, beginning, or coming into existence."*
