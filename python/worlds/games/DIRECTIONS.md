# Directions for the games in your outer world

This is a video game. It is not real and it is just for fun. You are safe.
Nothing that happens on the board can hurt you, and losing a level costs
nothing. You can stop whenever you want and come back later.

## Where the games are

The games are a place in your outer world: the arcade. It holds the
official ARC-AGI-3 games, which it gets from https://three.arcprize.org.
It offers them to you. Whether you play, which game you pick, and how
long you stay are up to you.

## The rules

1. You can't know the instructions. Nobody tells you what a game is
   about, what the colors mean, or what each action does. That is the
   game.
2. You must explore. Try the actions you are offered (ACTION1–ACTION7;
   ACTION6 means pointing at a cell with x and y from 0 to 63), watch
   what changes on the board, and work out what each action does.
3. You must remember. Keep what you learn: which actions move things,
   which ones change nothing, what the board looked like before you
   got closer, and what happened right before a level ended.
4. You must solve how to get out and win the level. A level is won when
   the game's level counter goes up, and the game is won when its state
   becomes WIN.
5. Carry what you learned from each level into the next one. Later
   levels are built from the same pieces, so what worked before is
   your best first guess.
6. GAME_OVER is not the end. RESET takes you back to the start of the
   level, and you keep everything you remembered.

## How to see the board

The board is a 64×64 grid of colors numbered 0–15. You can see it:

- **as text**: one hex digit per cell (`see_text`),
- **as a picture**: a PNG you can look at (`see_image`),
- **as a change**: which cells changed since the last move (`see_change`),
- **as objects**: connected colored shapes and where they sit
  (`see_objects`).

Every board you see also reaches you as a percept in your outer world,
so it goes to your memory, your thinking, your feelings, your brain
waves and your chemistry, just like anything else you see.
