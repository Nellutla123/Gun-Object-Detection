document.addEventListener("DOMContentLoaded", () => {
  // Element selections
  const imageUpload = document.getElementById("imageUpload");
  const predictButton = document.getElementById("predictButton");
  const uploadArea = document.getElementById("uploadArea");
  const uploadPreview = document.getElementById("uploadPreview");
  const previewImage = document.getElementById("previewImage");
  const changeImageBtn = document.getElementById("changeImageBtn");
  const originalImage = document.getElementById("originalImage");
  const processedImage = document.getElementById("processedImage");
  const statusMessage = document.getElementById("statusMessage");
  const resultsContainer = document.getElementById("resultsContainer");
  const thresholdInput = document.getElementById("thresholdInput");
  const thresholdLabel = document.getElementById("thresholdLabel");
  const detectedCount = document.getElementById("detectedCount");
  const detectionsList = document.getElementById("detectionsList");

  let currentFile = null;

  // Navigation smooth scrolling
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href");
      const targetSection = document.querySelector(targetId);
      if (targetSection) {
        targetSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  // Update active navigation link on scroll
  window.addEventListener("scroll", () => {
    const sections = document.querySelectorAll("section[id]");
    const scrollPosition = window.scrollY + 100;

    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      const sectionId = section.getAttribute("id");
      const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

      if (
        scrollPosition >= sectionTop &&
        scrollPosition < sectionTop + sectionHeight
      ) {
        document
          .querySelectorAll(".nav-link")
          .forEach((link) => link.classList.remove("active"));
        if (navLink) navLink.classList.add("active");
      }
    });
  });

  // File upload handling
  uploadArea.addEventListener("click", () => {
    imageUpload.click();
  });

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  });

  imageUpload.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  });

  changeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageUpload.click();
  });

  function handleFileSelect(file) {
    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type)) {
      showStatusMessage(
        "Please select a valid image file (JPEG, JPG, or PNG)",
        "error"
      );
      return;
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      showStatusMessage("File size must be less than 10MB", "error");
      return;
    }

    currentFile = file;
    resetDetections();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      originalImage.src = e.target.result;

      // Hide upload content, show preview
      uploadArea.querySelector(".upload-content").style.display = "none";
      uploadPreview.style.display = "block";

      // Enable predict button
      predictButton.disabled = false;

      // Hide previous results
      resultsContainer.style.display = "none";
      clearStatusMessage();
    };
    reader.readAsDataURL(file);
  }

  thresholdInput?.addEventListener("input", () => {
    thresholdLabel.textContent = Number(thresholdInput.value).toFixed(2);
  });

  // Prediction handling
  predictButton.addEventListener("click", async () => {
    if (!currentFile) {
      showStatusMessage("Please upload an image first.", "error");
      return;
    }

    // Update UI for loading state
    predictButton.disabled = true;
    predictButton.querySelector("span").style.display = "none";
    predictButton.querySelector(".btn-loading").style.display = "block";
    showStatusMessage("Analyzing image... Please wait.", "loading");
    resultsContainer.style.display = "none";

    const formData = new FormData();
    formData.append("file", currentFile);
    const threshold = thresholdInput
      ? Number(thresholdInput.value).toFixed(2)
      : "0.50";

    try {
      const response = await fetch(
        `/predict/json/?score_threshold=${threshold}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (response.ok) {
        const data = await response.json();
        if (data.image) {
          processedImage.src = data.image;
        }
        updateDetections(data);

        // Show results
        showResults();
        showStatusMessage(
          "Analysis complete! Results displayed above.",
          "success"
        );

        // Scroll to results
        setTimeout(() => {
          resultsContainer.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }, 500);
      } else {
        let errorMessage = "An error occurred during analysis.";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          errorMessage = `Error: ${response.status} ${response.statusText}`;
        }
        resetDetections();
        showStatusMessage(errorMessage, "error");
      }
    } catch (error) {
      console.error("Error during prediction:", error);
      resetDetections();
      showStatusMessage(
        "Network error occurred. Please check your connection and try again.",
        "error"
      );
    } finally {
      // Reset button state
      predictButton.disabled = false;
      predictButton.querySelector("span").style.display = "block";
      predictButton.querySelector(".btn-loading").style.display = "none";
    }
  });

  function updateDetections(data) {
    const count = data?.count ?? 0;
    detectedCount.textContent = count;

    const items = Array.isArray(data?.detections) ? data.detections : [];
    detectionsList.innerHTML = "";

    if (!items.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent =
        count === 0 ? "No guns detected." : "Detection data unavailable.";
      detectionsList.appendChild(li);
      return;
    }

    items.forEach((det, index) => {
      const li = document.createElement("li");
      const labelSpan = document.createElement("span");
      labelSpan.textContent = `${det.label || "Gun"} • ${
        Math.round((det.score ?? 0) * 100) / 100
      }`;

      const boxSpan = document.createElement("span");
      boxSpan.className = "box";
      if (Array.isArray(det.box)) {
        const [x1, y1, x2, y2] = det.box.map((n) => Math.round(n));
        boxSpan.textContent = `#${index + 1} [${x1}, ${y1}, ${x2}, ${y2}]`;
      } else {
        boxSpan.textContent = `#${index + 1}`;
      }

      li.appendChild(labelSpan);
      li.appendChild(boxSpan);
      detectionsList.appendChild(li);
    });
  }

  function resetDetections() {
    detectedCount.textContent = "0";
    detectionsList.innerHTML =
      '<li class="empty">Upload an image to view detection metadata.</li>';
    processedImage.src = "";
  }

  function showResults() {
    // Show results container
    resultsContainer.style.display = "block";
  }

  function showStatusMessage(message, type = "info") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = "block";

    // Auto-hide success messages after 5 seconds
    if (type === "success") {
      setTimeout(() => {
        clearStatusMessage();
      }, 5000);
    }
  }

  function clearStatusMessage() {
    statusMessage.style.display = "none";
    statusMessage.textContent = "";
    statusMessage.className = "status-message";
  }

  // Add some interactive animations
  function addInteractiveAnimations() {
    // Parallax effect for hero section
    window.addEventListener("scroll", () => {
      const scrolled = window.pageYOffset;
      const parallax = document.querySelector(".particles-bg");
      if (parallax) {
        parallax.style.transform = `translateY(${scrolled * 0.5}px)`;
      }
    });

    // Add intersection observer for animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
        }
      });
    }, observerOptions);

    // Observe elements for animation
    document
      .querySelectorAll(".feature, .info-card, .image-card")
      .forEach((el) => {
        el.style.opacity = "0";
        el.style.transform = "translateY(20px)";
        el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
        observer.observe(el);
      });
  }

  // Initialize animations
  addInteractiveAnimations();

  // Add loading animation to detect button
  function addButtonLoadingAnimation() {
    const btn = predictButton;
    btn.addEventListener("mouseenter", () => {
      if (!btn.disabled) {
        btn.style.transform = "translateY(-2px)";
      }
    });

    btn.addEventListener("mouseleave", () => {
      if (!btn.disabled) {
        btn.style.transform = "translateY(0)";
      }
    });
  }

  addButtonLoadingAnimation();
});
