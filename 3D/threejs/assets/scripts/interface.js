// Scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
const controls = new THREE.OrbitControls( camera, renderer.domElement );
camera.position.z = 5;

// Chargement des données d'elevation
var img = new Image();
var data_tab;
img.src = 'assets/images/perlin.png';
img.onload = function() {
	dessiner(this);
  };


// Geometry Plan
geometry = new THREE.PlaneGeometry( 10, 10 ,img.height-1,img.width-1);
texture = THREE.ImageUtils.loadTexture( "assets/images/texture_finale.png" );
material = new THREE.MeshLambertMaterial({ map : texture ,side:THREE.DoubleSide,wireframe:false});
plane = new THREE.Mesh( geometry, material );
scene.add( plane );

// Light
ambiantlight = new THREE.AmbientLight(0xcccccc,1);
scene.add(ambiantlight);

function animate() {
	//En boucle 60 fois par seconde:
	requestAnimationFrame( animate );

	//cube.rotation.x += 0.01;
	//cube.rotation.y += 0.01;

	controls.update();
	renderer.render( scene, camera );
};

function dessiner(img) {
	const canvas = createElementNS('canvas');
	canvas.height = img.height;
	canvas.width = img.width;
	var ctx = canvas.getContext('2d')
	ctx.drawImage(img, 0,0);
	data_tab = ctx.getImageData(0, 0, img.width, img.height);
}

function elevation(data_tab,value=3) {
    for (let i= 0; i < plane.geometry.attributes.position.count; i++) {
        //plane.geometry.attributes.position.setZ(i,Math.random()*value+offset);
		plane.geometry.attributes.position.setZ(i,(data_tab.data[i*4]/255)*value);
    }
    plane.geometry.attributes.position.needsUpdate = true;
}

//elevation(data_tab);
animate();