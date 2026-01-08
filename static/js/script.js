// Sidebar toggle functionality
document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");

  function toggleSidebar() {
    sidebar.classList.toggle("show");
    sidebarBackdrop.classList.toggle("show");
  }

  sidebarToggle.addEventListener("click", toggleSidebar);
  sidebarBackdrop.addEventListener("click", toggleSidebar);

  // Close sidebar when clicking on a link (mobile)
  sidebar.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", function () {
      if (window.innerWidth < 768) {
        toggleSidebar();
      }
    });
  });

  // Handle window resize
  window.addEventListener("resize", function () {
    if (window.innerWidth >= 768) {
      sidebar.classList.remove("show");
      sidebarBackdrop.classList.remove("show");
    }
  });
});
