import { useGLTF } from "@react-three/drei";
import { useRef } from "react";

export default function CHRACTER(props) {
  const ref = useRef();
  const { scene } = useGLTF("/CHRACTER.glb");

  return (
    <group
      ref={ref}
      position={[0, -1.2, 0]}     // lift model a bit
      rotation={[0, Math.PI, 0]}  // face the camera
    //   scale={[1.3, 1.3, 1.3]}     // adjust size
      {...props}
    >
      <primitive object={scene} />
    </group>
  );
}