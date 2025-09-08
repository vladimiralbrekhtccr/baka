# Japanese Phrases Collection 🇯🇵

<div class="app-container">
  <nav class="sidebar">
    <div class="sidebar-header">
      <h3>📚 Sections</h3>
    </div>
    
    <div class="sidebar-search">
      <input type="text" id="searchBox" placeholder="🔍 Search phrases..." />
    </div>
    
    <div class="sidebar-controls">
      <button onclick="toggleRandomPhrase()" class="btn btn-green">🎲 Random Phrase</button>
      <button onclick="toggleAll()" class="btn btn-blue">📖 Toggle All</button>
    </div>
    
    <ul class="section-list">
      <li><a href="#daily-life" onclick="scrollToSection('daily-life')">🗣️ Daily Life</a></li>
      <li><a href="#greetings" onclick="scrollToSection('greetings')">👋 Greetings</a></li>
      <li><a href="#emotions" onclick="scrollToSection('emotions')">😊 Emotions</a></li>
      <li><a href="#anime-manga" onclick="scrollToSection('anime-manga')">🎌 Anime/Manga</a></li>
    </ul>
    
    <div class="sidebar-stats">
      <div class="stat-item">
        <span class="stat-number" id="totalPhrases">1</span>
        <span class="stat-label">Total Phrases</span>
      </div>
    </div>
  </nav>

  <main class="main-content">
    <div id="searchResults"></div>

    <section id="daily-life" class="phrase-section">
      <h2>🗣️ Daily Life</h2>

<details class="phrase-item">
<summary><strong>Dare ga baka yo? Jikken-dai ni natte kara iinasai. Tsugi wa yasashiku ne.</strong></summary>

**Romaji:** Dare ga baka yo? Jikken-dai ni natte kara iinasai. Tsugi wa yasashiku ne.

**Translation:** Who's the idiot? Say that after you become a test subject. Be gentle next time.

**Context:** A playful or teasing response, possibly from anime/manga

**Grammar Notes:**
- だれ (dare) = who
- ばか (baka) = idiot/fool  
- 実験台 (jikken-dai) = test subject
- やさしく (yasashiku) = gently/kindly

</details>
    </section>

    <section id="greetings" class="phrase-section">
      <h2>👋 Greetings</h2>
      <p class="section-placeholder">Add greeting phrases here using the template below.</p>
    </section>

    <section id="emotions" class="phrase-section">
      <h2>😊 Emotions</h2>
      <p class="section-placeholder">Add emotional expressions here using the template below.</p>
    </section>

    <section id="anime-manga" class="phrase-section">
      <h2>🎌 Anime/Manga</h2>
      <p class="section-placeholder">Add anime/manga phrases here using the template below.</p>
    </section>

    <section id="how-to-add" class="phrase-section">
      <h2>➕ How to Add New Phrases</h2>

To add new phrases, follow this structure:

```markdown
<details class="phrase-item">
<summary><strong>[Japanese text]</strong></summary>

**Romaji:** [Romanized pronunciation]

**Translation:** [English translation]

**Context:** [When/where it's used]

**Grammar Notes:**
- [Key vocabulary and grammar points]

</details>
```
    </section>
  </main>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const phrases = document.querySelectorAll('.phrase-item');
    const searchBox = document.getElementById('searchBox');
    const searchResults = document.getElementById('searchResults');
    
    // Search functionality
    searchBox.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        searchResults.innerHTML = '';
        
        if (query.length < 2) {
            phrases.forEach(phrase => phrase.style.display = 'block');
            return;
        }
        
        let hasResults = false;
        phrases.forEach(phrase => {
            const text = phrase.textContent.toLowerCase();
            if (text.includes(query)) {
                phrase.style.display = 'block';
                hasResults = true;
            } else {
                phrase.style.display = 'none';
            }
        });
        
        if (!hasResults) {
            searchResults.innerHTML = '<p style="color: #666; text-align: center; margin: 20px;">No phrases found matching "' + query + '"</p>';
        }
    });
    
    // Random phrase functionality
    window.toggleRandomPhrase = function() {
        phrases.forEach(phrase => phrase.removeAttribute('open'));
        const randomIndex = Math.floor(Math.random() * phrases.length);
        phrases[randomIndex].setAttribute('open', '');
        phrases[randomIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    
    // Toggle all phrases
    window.toggleAll = function() {
        const allOpen = Array.from(phrases).every(phrase => phrase.hasAttribute('open'));
        phrases.forEach(phrase => {
            if (allOpen) {
                phrase.removeAttribute('open');
            } else {
                phrase.setAttribute('open', '');
            }
        });
    };
    
    // Scroll to section function
    window.scrollToSection = function(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Update active sidebar link
            document.querySelectorAll('.section-list a').forEach(link => link.classList.remove('active'));
            document.querySelector(`a[href="#${sectionId}"]`).classList.add('active');
        }
    };
    
    // Update phrase count
    function updatePhraseCount() {
        const totalPhrases = document.querySelectorAll('.phrase-item').length;
        document.getElementById('totalPhrases').textContent = totalPhrases;
    }
    updatePhraseCount();
    
    // Add some styling
    const style = document.createElement('style');
    style.textContent = \`
        * {
            box-sizing: border-box;
        }
        
        .app-container {
            display: flex;
            min-height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .sidebar {
            width: 280px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        .sidebar-header h3 {
            margin: 0 0 20px 0;
            font-size: 1.4em;
            text-align: center;
        }
        
        .sidebar-search {
            margin-bottom: 20px;
        }
        
        .sidebar-search input {
            width: 100%;
            padding: 12px 15px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 14px;
        }
        
        .sidebar-search input::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        .sidebar-search input:focus {
            outline: none;
            background: rgba(255,255,255,0.3);
            box-shadow: 0 0 0 2px rgba(255,255,255,0.3);
        }
        
        .sidebar-controls {
            margin-bottom: 25px;
        }
        
        .btn {
            width: 100%;
            padding: 10px 15px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            margin-bottom: 8px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-green {
            background: #4CAF50;
            color: white;
        }
        
        .btn-blue {
            background: #2196F3;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .section-list {
            list-style: none;
            padding: 0;
            margin: 0 0 25px 0;
        }
        
        .section-list li {
            margin-bottom: 5px;
        }
        
        .section-list a {
            display: block;
            padding: 12px 15px;
            color: rgba(255,255,255,0.9);
            text-decoration: none;
            border-radius: 15px;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        
        .section-list a:hover,
        .section-list a.active {
            background: rgba(255,255,255,0.2);
            color: white;
            transform: translateX(5px);
        }
        
        .sidebar-stats {
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 20px;
            text-align: center;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #FFD700;
        }
        
        .stat-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .main-content {
            margin-left: 280px;
            padding: 30px 40px;
            flex: 1;
            background: #f8f9fa;
        }
        
        .phrase-section {
            margin-bottom: 40px;
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .phrase-section h2 {
            margin-top: 0;
            color: #333;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .section-placeholder {
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 2px dashed #ddd;
        }
        
        .phrase-item {
            margin: 20px 0;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .phrase-item:hover {
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .phrase-item summary {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 18px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .phrase-item summary:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .phrase-item[open] summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .phrase-item > div, .phrase-item > p {
            padding: 20px;
            background: white;
        }
        
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                position: relative;
                height: auto;
            }
            
            .main-content {
                margin-left: 0;
                padding: 20px;
            }
            
            .app-container {
                flex-direction: column;
            }
        }
    \`;
    document.head.appendChild(style);
});
</script>

---

*Last updated: $(date)*

> **Tip:** Click on any phrase to expand it and see details. Use the search box to find specific phrases, or click "Random Phrase" for practice!