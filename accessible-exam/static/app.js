/**
 * Single-page dashboard: switch between mode selection and exam views
 */
(function () {
  function showView(viewId) {
    document.querySelectorAll(".view").forEach(function (el) {
      el.classList.remove("view-active");
    });
    var el = document.getElementById("view-" + viewId);
    if (el) el.classList.add("view-active");
  }

  document.addEventListener("click", function (e) {
    var goto = e.target.getAttribute("data-goto");
    if (goto) {
      e.preventDefault();
      showView(goto);
    }
  });
})();

