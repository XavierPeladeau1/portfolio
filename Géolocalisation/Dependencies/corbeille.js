var overlayMaps = {
    "Vidéotron - 3G": vl_3gLayer,
    "Vidéotron - LTE": vl_lteLayer,
    "Rogers - 3G": rw_3gLayer
};


L.control.layers(null, overlayMaps, {
    position: "bottomright",
    collapsed: false
}).addTo(mymap)
L.control.zoom({
    position: "bottomright"
}).addTo(mymap)


//Icône custom Marker
var PointRouge = L.icon({
    iconUrl: 'Images/fg.png',
    shadowUrl: 'Images/bg.png',

    iconSize: [20, 30], // size of the icon
    shadowSize: [10, 15], // size of the shadow
    iconAnchor: [10, 30], // point of the icon which will correspond to marker's location
    shadowAnchor: [4, 19], // the same for the shadow
    popupAnchor: [0, -20]
});;



mymap.pm.enableDraw("Polygon", {
    snappable: true,
    snapDistance: 20,
    templineStyle: {
        color: "red"
    },
    hintlineStyle: {
        color: "red",
        dashArray: [5, 5],
    },
});
mymap.pm.disableDraw("Polygon")



mymap.pm.enableDraw("Line", {
    snappable: true,
    snapDistance: 20,
    templineStyle: {
        color: "red"
    },
    hintlineStyle: {
        color: "red",
        dashArray: [5, 5],
    },
});
mymap.pm.disableDraw("Line")



function addPoly(e) {
    switch (e.properties.tech) {
        case "vl_3g":
            vl_3gLayer.addData(e)
            break;
        case "vl_lte":
            vl_lteLayer.addData(e)
            break;
        case "rw_3g":
            rw_3gLayer.addData(e)
            break;
        default:
            layerGroup.addData(e)
    }
    // marker.bindPopup("Numéro de cellule: <b><div class='text-center'>" + e.cellule.toString() + "</div>")
    listeantennes.push({cell: e.properties.cell_id, lac: e.properties.lac_tac})
}

function filterSize() {
    var maxsize = parseInt($("#area_slider").val())
    cellgroup.eachLayer(function (l) {
        l.eachLayer(function (m) {
            console.log(maxsize)
            console.log(m.feature.geometry.properties.aire)
            if (maxsize < m.feature.geometry.properties.aire) {
                l.removeLayer(m)
            }
        });
    });
}


function parseJSONResponse(data) {
    for (i in data) {
        thispoly = JSON.parse(data[i].Polygone)
        thispoly['properties'] = {
            heure: data[i].Heure,
            color: polygoneColor(data[i].Tech),
            tech: data[i].Tech
        };
        layerGroup.addData(thispoly)
    }
}