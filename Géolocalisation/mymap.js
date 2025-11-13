/* ------------------------------------------------------------ Initialisation ---------------------------------------------------- */
/* ------------------------------------------------------------ Initialisation ---------------------------------------------------- */
/* ------------------------------------------------------------ Initialisation ---------------------------------------------------- */
//Initialisation map


const basemap = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
        '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
        'Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    id: 'mapbox.streets'
});

//Drawing layer
var layerGroup = L.geoJSON(null, {
    style: jsonstyle
});

var mymap = L.map('mapid', {
    zoomControl: false,
    center: {
        lat: 45.692271744965986,
        lng: -73.81027221679689
    },
    zoom: 10,
    layers: [basemap, layerGroup],
    maxBounds: [
        [44.2215, -86.781],
        [51.0413, -65.6872]
    ]
});

$.getJSON("totalCoverage.json",
    function (data) {
        L.geoJSON(data, {
            invert: true,
            style: function () {
                return {
                    fillOpacity: 0.25,
                    opacity: 0,
                    fillColor: "grey",
                }
            },
            pane: "tilePane"
        }).addTo(mymap)
    }
);


L.control.zoom({
    position: "bottomright"
}).addTo(mymap)




//JSON polygones style
function polygoneColor(e) {
    return e == "rw_3g" ? '#d80b00' :
        e == "vl_3g" ? '#505050' :
        e == "vl_lte" ? '#ffc000' :
        '#ff00e7';
}

function realtimeOpacity(h, i) {
    diff = i - h
    if (h < i && diff < 1800) {
        return (1 - diff / 1800) * 0.8
    } else {
        return 0
    }
}

function jsonstyle(poly) {
    return {
        fillColor: poly.geometry.properties.color,
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.4
    };
}

function realtimestyle(poly) {
    return {
        fillColor: poly.geometry.properties.color,
        weight: 1.5,
        opacity: realtimeOpacity(poly.geometry.properties.heure, parseInt($('#timeline_time').val())),
        color: 'black',
        fillOpacity: realtimeOpacity(poly.geometry.properties.heure, parseInt($('#timeline_time').val()))
    };
}



mymap.on("pm:create", function (e) {
    $(".btn").removeClass("active")
    mymap.removeLayer(e.layer)
    layerGroup.addLayer(e.layer)
    var feature = JSON.stringify(e.layer.toGeoJSON().geometry)
    QueryData(feature)
});




//Visualisation
//Visualisation
//Visualisation


function QueryData(geo_forme_json) {
    $("#loading").css("display", "flex")
    $.post(
        "backend-server-url/geoloc-cells", {
            geo_forme_json: geo_forme_json
        },
        function (data) {
            var thisline
            for (i in data) {
                thisline = data[i]
                thispoly = thisline.polygone
                thispoly = JSON.parse(thispoly)
                thispoly['properties'] = {
                    lac_tac: thisline.lac_tac,
                    tech: thisline.tech,
                    color: polygoneColor(thisline.tech),
                    cell_id: thisline.cell_id,
                };
                console.log(thisline)
                layerGroup.addData(thispoly)
            }
            $("#loading").css("display", "none")
            GenCellListe()
        });
}



function GetTrackingData(imei, date) {
    $("#loading").css("display", "flex")
    $.post(
        "backend-server-url/geoloc-imei", {
            imei: imei,
            date: date
        },
        function (data) {
            for (i in data) {
                thispoly = data[i].Polygone
                thispoly = JSON.parse(thispoly)
                thispoly['properties'] = {
                    heure: data[i].Heure,
                    color: polygoneColor(data[i].Tech),
                    tech: data[i].Tech
                };
                layerGroup.addData(thispoly)
            }
            if (data.length < 1) {
                alert("Aucune activité trouvée pour ces paramètres.")
            } else {
                $("#menu").slideToggle("slow")
                $("#results_realtime").slideToggle("slow")
                layerGroup.setStyle(realtimestyle)
            }
            $("#loading").css("display", "none")
            $("#timeline_contacts").text(data.length)
        });
}
