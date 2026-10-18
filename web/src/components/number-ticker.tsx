"use client";

import { useEffect } from "react";
import {
  animate,
  useMotionValue,
  useReducedMotion,
  useTransform,
  motion,
} from "motion/react";

/** Animated number that counts to `value` on mount (metric cards).
 *  `format` maps the raw value to its display string. Reduced motion:
 *  shows the final value immediately. */
export function NumberTicker({
  value,
  format = (v: number) => v.toFixed(0),
  duration = 1.1,
  className,
}: {
  value: number;
  format?: (v: number) => string;
  duration?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(reduce ? value : 0);
  const text = useTransform(mv, (v) => format(v));

  useEffect(() => {
    if (reduce) {
      mv.set(value);
      return;
    }
    const controls = animate(mv, value, { duration, ease: [0.22, 1, 0.36, 1] });
    return controls.stop;
  }, [value, duration, mv, reduce]);

  return <motion.span className={className}>{text}</motion.span>;
}
