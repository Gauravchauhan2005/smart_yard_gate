/**
 * AI-Based Smart Yard Gate Automation System
 * Core Client-Side Application Script
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Mobile Sidebar Toggle
    const sidebarToggle = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener("click", function () {
            sidebar.classList.toggle("open");
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener("click", function (event) {
            if (
                window.innerWidth < 992 &&
                !sidebar.contains(event.target) &&
                !sidebarToggle.contains(event.target) &&
                sidebar.classList.contains("open")
            ) {
                sidebar.classList.remove("open");
            }
        });
    }

    // 2. Live Topbar Clock (UTC / Local)
    const liveClock = document.getElementById("liveClock");
    if (liveClock) {
        function updateClock() {
            const now = new Date();
            const timeString = now.toLocaleTimeString("en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            });
            liveClock.textContent = `${timeString} LOCAL`;
        }
        updateClock();
        setInterval(updateClock, 1000);
    }
});
