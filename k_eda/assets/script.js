/* ============================================================
   K-EDA — Report Interactivity
   ============================================================ */

// ── Sidebar Navigation Active Tracking ──
(function() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.report-section');

    // Tab isolation / Dashboard logic on nav click
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Set active class on nav
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            const sectionId = this.getAttribute('data-section');

            if (sectionId === 'all') {
                // Dashboard mode: show all sections
                sections.forEach(sec => sec.style.display = 'block');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                // Tab mode: isolate one section
                sections.forEach(sec => {
                    if (sec.getAttribute('data-section') === sectionId) {
                        sec.style.display = 'block';
                        // Trigger Plotly resize for charts inside this newly visible section
                        const charts = sec.querySelectorAll('.plotly-chart');
                        charts.forEach(chart => {
                            if (window.Plotly && chart.data) {
                                Plotly.Plots.resize(chart);
                            }
                        });
                    } else {
                        sec.style.display = 'none';
                    }
                });
                window.scrollTo({ top: 0, behavior: 'instant' });
            }

            // Close mobile sidebar
            const sidebar = document.getElementById('sidebar');
            if (sidebar) sidebar.classList.remove('open');
        });
    });

    // We no longer need the IntersectionObserver because the click event manages the active state.
})();

// ── Fade-in Animation on Scroll ──
(function() {
    var sections = document.querySelectorAll('.report-section');
    sections.forEach(function(section, index) {
        section.style.animationDelay = (index * 0.05) + 's';
    });
})();

// ── Tab Switching ──
function switchTab(btn, tabId) {
    // Deactivate all tabs in the same container
    var container = btn.closest('.tab-container');
    if (!container) return;

    container.querySelectorAll('.tab-btn').forEach(function(b) {
        b.classList.remove('active');
    });
    container.querySelectorAll('.tab-content').forEach(function(c) {
        c.style.display = 'none';
    });

    btn.classList.add('active');
    var target = document.getElementById(tabId);
    if (target) {
        target.style.display = 'block';
        // Trigger Plotly resize for charts that were hidden
        var charts = target.querySelectorAll('.plotly-chart');
        charts.forEach(function(chart) {
            if (window.Plotly && chart.data) {
                Plotly.Plots.resize(chart);
            }
        });
    }
}

// ── Mobile Sidebar Toggle ──
function toggleSidebar() {
    var sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
}

// Close sidebar on outside click (mobile)
document.addEventListener('click', function(e) {
    var sidebar = document.getElementById('sidebar');
    var menuBtn = document.getElementById('mobile-menu-btn');
    if (sidebar && sidebar.classList.contains('open')) {
        if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    }
});

// ── Window resize: trigger Plotly resize ──
var resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
        document.querySelectorAll('.plotly-chart').forEach(function(chart) {
            if (window.Plotly && chart.data) {
                Plotly.Plots.resize(chart);
            }
        });
    }, 200);
});

// ── Variable Card Dropdown Switcher ──
function showVariableCard(colId) {
    var cards = document.querySelectorAll('.variable-card');
    cards.forEach(function(card) {
        card.style.display = 'none';
    });
    var targetCard = document.getElementById('var-' + colId);
    if (targetCard) {
        targetCard.style.display = 'block';
        // Resize charts inside the newly shown card
        var charts = targetCard.querySelectorAll('.plotly-chart');
        charts.forEach(function(chart) {
            if (window.Plotly && chart.data) {
                Plotly.Plots.resize(chart);
            }
        });
    }
}

// ── Emoji Polishing ──
(function() {
    function polishEmojis() {
        // 1. Sidebar Nav Links
        document.querySelectorAll('.nav-link').forEach(el => {
            const text = el.innerHTML;
            const emojiRegex = /^([\p{Emoji_Presentation}\p{Emoji}\u200d]+)\s*/u;
            const match = text.match(emojiRegex);
            if (match) {
                const emoji = match[1];
                const rest = text.substring(match[0].length);
                el.innerHTML = `<span class="emoji-badge">${emoji}</span><span class="nav-text">${rest}</span>`;
            }
        });

        // 2. Card Titles and Section Titles (if any start with emojis)
        document.querySelectorAll('.card-title, .section-title').forEach(el => {
            const text = el.innerHTML;
            const emojiRegex = /^([\p{Emoji_Presentation}\p{Emoji}\u200d]+)\s*/u;
            const match = text.match(emojiRegex);
            if (match) {
                const emoji = match[1];
                const rest = text.substring(match[0].length);
                el.innerHTML = `<span class="title-emoji-badge">${emoji}</span><span class="title-text">${rest}</span>`;
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', polishEmojis);
    } else {
        polishEmojis();
    }
})();

