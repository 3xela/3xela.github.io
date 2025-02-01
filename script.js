// Check for saved theme preference, otherwise use system preference
const getPreferredTheme = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };
  
  // Function to set theme
  const setTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    // Update toggle button text
    const toggleText = document.querySelector('.theme-toggle-text');
    toggleText.textContent = theme === 'dark' ? '☀️' : '🌙';
  };
  
  // Initialize theme
  setTheme(getPreferredTheme());
  
  // Add click handler to toggle button
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    setTheme(currentTheme === 'dark' ? 'light' : 'dark');
  });
  
  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
      setTheme(e.matches ? 'dark' : 'light');
    }
  });