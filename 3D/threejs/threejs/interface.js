
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

geometry = new THREE.BoxGeometry(3,2,1);
material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
cube = new THREE.Mesh( geometry, material );
//scene.add(cube);

geometry = new THREE.PlaneGeometry( 10, 10 ,10,10);
texture = THREE.ImageUtils.loadTexture( "texture_finale.jpg" );
//material = new THREE.MeshBasicMaterial( {color: 0xffff00, side: THREE.DoubleSide} );
material = new THREE.MeshLambertMaterial({ map : texture ,side:THREE.DoubleSide,wireframe:true});
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

	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

	controls.update();

	renderer.render( scene, camera );
};

function randomize_elevation(value=0.2, offset=0) {
    for (let i= 0; i < plane.geometry.attributes.position.length/3; i++) {
        plane.geometry.attributes.position.setZ(i,Math.random()*value+offset);
    }
    plane.geometry.attributes.position.needsUpdate = true;
}
animate();