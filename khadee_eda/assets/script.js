/* ============================================================
   Khadee EDA — Report Interactivity
   ============================================================ */

// ── Sidebar Navigation Active Tracking ──
(function() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.report-section');

    // Smooth scroll on nav click
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            // Close mobile sidebar
            document.getElementById('sidebar').classList.remove('open');
        });
    });

    // Intersection Observer for active nav highlighting
    if (sections.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.getAttribute('data-section');
                    navLinks.forEach(function(link) {
                        link.classList.remove('active');
                        if (link.getAttribute('data-section') === sectionId) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, {
            rootMargin: '-20% 0px -70% 0px',
            threshold: 0
        });

        sections.forEach(function(section) {
            observer.observe(section);
        });
    }

    // Set first nav link active by default
    if (navLinks.length > 0) {
        navLinks[0].classList.add('active');
    }
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

