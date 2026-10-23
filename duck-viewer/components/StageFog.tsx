"use client";

// The stage's fog, sized from the view rather than nailed to fixed metres.
// Both pages mount it in place of a bare <fog>; the arithmetic — and the
// measurement that moved it — is lib/camera.fogRange.

import { useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { fogRange } from "@/lib/camera";

export interface StageFogProps {
  /** Fog colour — the page's background, so the fade reads as depth. */
  color: string;
  /** The page's tuned band. `near` is a floor: a close-up never fogs more
   *  than it did before. */
  near: number;
  far: number;
  /** Cap on the far plane, so the band can't walk past the floor it exists
   *  to fade out. */
  farMax?: number;
  /** Orbit target to measure from when OrbitControls hasn't mounted yet. */
  target?: readonly [number, number, number];
}

export default function StageFog({ color, near, far, farMax, target }: StageFogProps) {
  const fog = useRef<THREE.Fog>(null);
  const controls = useThree((s) => s.controls) as { target?: THREE.Vector3 } | null;
  const fallback = useMemo(() => new THREE.Vector3(...(target ?? [0, 0.12, 0])), [target]);
  // Per-frame, not per-render: the orbit radius changes on every drag and
  // wheel tick, none of which pass through React. near/far are plain uniforms
  // on the same Fog instance, so writing them costs nothing and recompiles
  // no shaders (only the fog's PRESENCE and TYPE are material defines).
  useFrame(({ camera }) => {
    const f = fog.current;
    if (!f) return;
    const t = controls?.target ?? fallback;
    const r = fogRange(camera.position.distanceTo(t), near, far, farMax);
    f.near = r.near;
    f.far = r.far;
  });
  return <fog ref={fog} attach="fog" args={[color, near, far]} />;
}
