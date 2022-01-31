const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );


// geometry = new THREE.BoxGeometry(3,2,1);
// material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
// cube = new THREE.Mesh( geometry, material );
//scene.add(cube);
var img = new Image();
var data_tab;
img.src = 'perlin.png';
img.onload = function() {
	dessiner(this);
  };
function dessiner(img) {
	const canvas = createElementNS('canvas');
	canvas.height = img.height;
	canvas.width = img.width;
	var ctx = canvas.getContext('2d')
	ctx.drawImage(img, 0,0);
	data_tab = ctx.getImageData(0, 0, img.width, img.height);
}
// yo = canvas.getContext('2d');
// yo.drawImage(img, 0, 0,img.width,img.height);
// imagedata = yo.getImageData(0, 0, img.width, img.height);
// imagedata.needsUpdate = true;

// Geometry Plan
geometry = new THREE.PlaneGeometry( 10, 10 ,img.height-1,img.width-1);
texture = THREE.ImageUtils.loadTexture( "texture_finale.png" );
material = new THREE.MeshLambertMaterial({ map : texture ,side:THREE.DoubleSide,wireframe:false});
plane = new THREE.Mesh( geometry, material );

scene.add( plane );

//const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.5 );
//scene.add( directionalLight );

ambiantlight = new THREE.AmbientLight(0xcccccc,1);
scene.add(ambiantlight);

const controls = new THREE.OrbitControls( camera, renderer.domElement );

camera.position.z = 5;

function animate() {
	//En boucle 60 fois par seconde:
	requestAnimationFrame( animate );

	//cube.rotation.x += 0.01;
	//cube.rotation.y += 0.01;

	controls.update();
	renderer.render( scene, camera );
};

function randomize_elevation(value=0.2, offset=0) {
	error = value
    for (let i= 0; i < plane.geometry.attributes.position.count; i++) {
        //plane.geometry.attributes.position.setZ(i,Math.random()*value+offset);
		plane.geometry.attributes.position.setZ(i,(data_tab.data[i*4]/255)*3);
    }
    plane.geometry.attributes.position.needsUpdate = true;
}
animate();