// Make project cards fully clickable
function initProjectCards() {
  const cards = document.querySelectorAll('.project-card');
  cards.forEach(function(card) {
    const link = card.querySelector('h3 a');
    if (link && !card.dataset.initialized) {
      card.dataset.initialized = 'true';
      card.addEventListener('click', function(e) {
        // Don't trigger if clicking directly on the link
        if (!e.target.closest('a')) {
          e.preventDefault();
          e.stopPropagation();
          window.location.href = link.href;
        }
      });
    }
  });
}

// Run on initial load
document.addEventListener('DOMContentLoaded', initProjectCards);

// Run on MkDocs instant navigation
document.addEventListener('DOMContentSwitch', initProjectCards);

// Also observe for dynamic content changes
if (typeof MutationObserver !== 'undefined') {
  const observer = new MutationObserver(initProjectCards);
  observer.observe(document.body, { childList: true, subtree: true });
}
