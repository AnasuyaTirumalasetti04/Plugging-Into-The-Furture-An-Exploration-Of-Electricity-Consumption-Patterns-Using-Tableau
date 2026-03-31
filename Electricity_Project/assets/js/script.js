var divElement = document.getElementById("vizVisualization");
var vizElement = divElement.getElementsByTagName("object")[0];

vizElement.style.width = "100%";
vizElement.style.height = "900px";

var scriptElement = document.createElement("script");
scriptElement.src = "https://public.tableau.com/javascripts/api/viz_v1.js";

vizElement.parentNode.insertBefore(scriptElement, vizElement);