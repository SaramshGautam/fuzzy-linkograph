// import embeddings lib
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0/dist/transformers.min.js";
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
const DIMENSION = 384;
// import react
const e = React.createElement;

/// global graph properties

const MOVE_TEXT_MODE = "FULL"; // or "FULL" or "INDEX"
const SHOULD_COLORIZE_LINKS = true;
const GRAPH_WIDTH = 1000;
const INIT_X = 10;
const INIT_Y = { FULL: 500, INDEX: 80, NONE: 60 }[MOVE_TEXT_MODE];
const MOVE_LINK_BAR_HEIGHT = 40; // how tall the forelink/backlink bars over each move should be
const MIN_LINK_STRENGTH = 0.2;
const SEGMENT_THRESHOLD = 1000 * 60 * 30; // 30 mins -> milliseconds
// const MOVE_TEXT_MODE = "INDEX"; // or "FULL"

// Explanations for hover tooltips in StatsBar
const METRIC_INFO = {
  "Link density":
    "Sum of all (thresholded & scaled) link weights divided by number of moves. Higher = more overall connectedness.",
  "Entropy (total)":
    "Backlink + Forelink + Horizonlink entropy. Higher = less predictable link structure across the graph.",
  "Entropy back":
    "Per-move backlink entropy: uses backlinkWeight / maxPossibleBacklinkWeight, summed across moves.",
  "Entropy fore":
    "Per-move forelink entropy: uses forelinkWeight / maxPossibleForelinkWeight, summed across moves.",
  "Entropy horizon":
    "Entropy of links across fixed time ‘horizons’ (pairs i and i+N) over all N. Captures longer-range structure.",
  "Near-copies":
    "Count of links ≥ 0.99 (treated as near-verbatim copies) that are skipped in actor-pair density stats.",
  "Max fore W":
    "Maximum forelinkWeight observed across moves after thresholding & scaling.",
  "Max back W":
    "Maximum backlinkWeight observed across moves after thresholding & scaling.",
  "Sim threshold":
    "MIN_LINK_STRENGTH: links below this cosine similarity are ignored; above it get scaled to [0..1].",
  "Segment gap":
    "If the time between consecutive moves ≥ this (in minutes), draw a vertical divider in the timeline.",
  "Colorized links":
    "If ON and multiple actors exist, link color blends depend on actor pairing and similarity strength.",
  "Text mode":
    "Move label style: FULL shows text; INDEX shows indices; NONE hides labels.",
};

// Map dataset path -> related artifact files (images/CSVs) shown in the sidebar
const ARTIFACTS_BY_DATASET = {
  "./data/teamx_v3_linked.json": {
    images: [
      "./data/outputs_v3/teamx_phase_confidence_timeline.png",
      "./data/outputs_v3/teamx_phase_confidence_lines.png",
    ],
    csvs: [
      "./data/outputs_v3/teamx_window_metrics_labeled.csv",
      "./data/outputs_v3/teamx_phase_confidence.csv",
    ],
  },

  "./data/teamx_v2_linked.json": {
    images: [
      "./data/outputs_v2_clip/teamx_phase_confidence_timeline.png",
      "./data/outputs_v2_clip/teamx_phase_confidence_lines.png",
    ],
    csvs: [
      "./data/outputs_v2_clip/teamx_window_metrics_labeled.csv",
      "./data/outputs_v2_clip/teamx_phase_confidence.csv",
    ],
  },
  "./data/teamx_V1_linked.json": {
    images: [
      "./data/outputs_v1_clip/teamx_phase_confidence_timeline.png",
      "./data/outputs_v1_clip/teamx_phase_confidence_lines.png",
    ],
    csvs: [
      "./data/outputs_v1_clip/teamx_window_metrics_labeled.csv",
      "./data/outputs_v1_clip/teamx_phase_confidence.csv",
    ],
  },
  "./data/teamx_session_10min_linked.json": {
    images: [
      "./data/outputs_v0_clip/teamx_phase_confidence_timeline.png",
      "./data/outputs_v0_clip/teamx_phase_confidence_lines.png",
    ],
    csvs: [
      "./data/outputs_v0_clip/teamx_window_metrics_labeled.csv",
      "./data/outputs_v0_clip/teamx_phase_confidence.csv",
    ],
  },
  "./data/team_roadtrip_session_10min_linked.json": {
    images: [
      "./data/outputs_roadtrip_v4/teamx_phase_confidence_timeline.png",
      "./data/outputs_roadtrip_v4/teamx_phase_confidence_lines.png",
    ],
    csvs: [
      "./data/outputs_roadtrip_v4/teamx_window_metrics_labeled.csv",
      "./data/outputs_roadtrip_v4/teamx_phase_confidence.csv",
    ],
  },
};

// Distinct colors per actor; extend if you have more actors.
const ACTOR_COLORS = {
  0: { r: 230, g: 73, b: 53 }, // red-ish
  1: { r: 32, g: 82, b: 204 }, // blue-ish
  2: { r: 20, g: 156, b: 88 }, // green-ish
};

// Fallback color if actor id is unknown
const DEFAULT_COLOR = { r: 120, g: 120, b: 120 };

function rgbToStr({ r, g, b }) {
  return `rgb(${r},${g},${b})`;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

// Blend two actor colors based on link strength (stronger => closer to target)
function blendColors(c1, c2, t) {
  const w = clamp01((t - MIN_LINK_STRENGTH) / (1 - MIN_LINK_STRENGTH));
  return {
    r: Math.round(c1.r + (c2.r - c1.r) * w),
    g: Math.round(c1.g + (c2.g - c1.g) * w),
    b: Math.round(c1.b + (c2.b - c1.b) * w),
  };
}

/// math utils

function sum(xs) {
  return xs.reduce((a, b) => a + b, 0);
}

function dotProduct(vectorA, vectorB) {
  let dotProd = 0;
  for (let comp = 0; comp < DIMENSION; comp++) {
    dotProd += vectorA[comp] * vectorB[comp];
  }
  return dotProd;
}

function magnitude(vector) {
  return Math.sqrt(dotProduct(vector, vector));
}

function cosineSimilarity(vectorA, vectorB) {
  const dotProd = dotProduct(vectorA, vectorB);
  const magProd = magnitude(vectorA) * magnitude(vectorB);
  return dotProd / magProd;
}

// Translate `num` from the scale bounded by `oldMin` and `oldMax`
// to the scale bounded by `newMin` and `newMax`.
function scale(num, [oldMin, oldMax], [newMin, newMax]) {
  const oldRange = oldMax - oldMin;
  const newRange = newMax - newMin;
  return ((num - oldMin) / oldRange) * newRange + newMin;
}

function colorForActor(actorId) {
  return ACTOR_COLORS.hasOwnProperty(actorId)
    ? ACTOR_COLORS[actorId]
    : DEFAULT_COLOR;
}

/// stats on a computed linkograph

// Given `linkStrengths`, a list of raw (unscaled) semantic similarity scores:
// 1. Threshold each semantic similarity value on `MIN_LINK_STRENGTH`.
// 2. Scale it from the range `[0, 1]` to the range `[MIN_LINK_STRENGTH, 1]`.
// 3. Sum up all the scaled values and return the sum.
function totalLinkWeight(linkStrengths) {
  return sum(
    linkStrengths
      .filter((n) => n >= MIN_LINK_STRENGTH)
      .map((n) => scale(n, [MIN_LINK_STRENGTH, 1], [0, 1]))
  );
}

function computeLinkDensityIndex(graph) {
  const overallLinkWeight = sum(
    Object.values(graph.links).map((linkSet) =>
      totalLinkWeight(Object.values(linkSet))
    )
  );
  graph.linkDensityIndex = overallLinkWeight / graph.moves.length;
}

function computeMoveWeights(graph) {
  for (let i = 0; i < graph.moves.length; i++) {
    const backlinkStrengths = Object.values(graph.links[i]);
    graph.moves[i].backlinkWeight = totalLinkWeight(backlinkStrengths);
    const forelinkStrengths = Object.values(graph.links).map(
      (linkSet) => linkSet[i] || 0
    );
    graph.moves[i].forelinkWeight = totalLinkWeight(forelinkStrengths);
  }
  // naïvely mark critical moves: top 3 link weights in each direction
  graph.moves
    .toSorted((a, b) => b.backlinkWeight - a.backlinkWeight)
    .slice(0, 3)
    .forEach((move) => {
      move.backlinkCriticalMove = true;
    });
  graph.moves
    .toSorted((a, b) => b.forelinkWeight - a.forelinkWeight)
    .slice(0, 3)
    .forEach((move) => {
      move.forelinkCriticalMove = true;
    });
  // calculate max forelink and backlink weights seen, for visual scaling
  graph.maxForelinkWeight = Math.max(
    ...graph.moves.map((move) => move.forelinkWeight)
  );
  graph.maxBacklinkWeight = Math.max(
    ...graph.moves.map((move) => move.backlinkWeight)
  );
}

function entropy(pOn, pOff) {
  const pOnPart = pOn > 0 ? -(pOn * Math.log2(pOn)) : 0;
  const pOffPart = pOff > 0 ? -(pOff * Math.log2(pOff)) : 0;
  return pOnPart + pOffPart;
}

function computeEntropy(graph) {
  // backlinks and forelinks
  for (let i = 0; i < graph.moves.length; i++) {
    // backlinks
    const maxPossibleBacklinkWeight = i;
    const backlinkPOn =
      graph.moves[i].backlinkWeight / maxPossibleBacklinkWeight;
    const backlinkPOff = 1 - backlinkPOn;
    graph.moves[i].backlinkEntropy = entropy(backlinkPOn, backlinkPOff);
    // forelinks
    const maxPossibleForelinkWeight = graph.moves.length - (i + 1);
    const forelinkPOn =
      graph.moves[i].forelinkWeight / maxPossibleForelinkWeight;
    const forelinkPOff = 1 - forelinkPOn;
    graph.moves[i].forelinkEntropy = entropy(forelinkPOn, forelinkPOff);
  }
  graph.backlinkEntropy = sum(graph.moves.map((move) => move.backlinkEntropy));
  graph.forelinkEntropy = sum(graph.moves.map((move) => move.forelinkEntropy));
  // horizonlinks
  // Each "horizon state" is the set of possible links between pairs of moves
  // that are N apart from each other. In traditional linkography, the one-link
  // horizon state at the max possible value of N is skipped, because it always
  // has entropy of exactly 0 (the singular link either exists or doesn't, and
  // is fully "predictable" either way). For fuzzy linkography, however, we
  // include this state in the calculation, because that link may be of some
  // unknown nonzero strength and therefore some unknown nonzero entropy value.
  graph.horizonlinkEntropy = 0;
  for (let horizon = 1; horizon < graph.moves.length; horizon++) {
    // get all pairs of move indexes (i,j) that are N apart
    const moveIndexPairs = [];
    for (let i = 0; i < graph.moves.length - horizon; i++) {
      moveIndexPairs.push([i, i + horizon]);
    }
    // get total link weight between these pairs
    const horizonlinkStrengths = moveIndexPairs.map(
      ([i, j]) => graph.links[j]?.[i] || 0
    );
    const horizonlinkWeight = totalLinkWeight(horizonlinkStrengths);
    // calculate entropy
    const horizonlinkPOn = horizonlinkWeight / moveIndexPairs.length;
    const horizonlinkPOff = 1 - horizonlinkPOn;
    graph.horizonlinkEntropy += entropy(horizonlinkPOn, horizonlinkPOff);
  }
  // sum them all up
  graph.entropy =
    graph.backlinkEntropy + graph.forelinkEntropy + graph.horizonlinkEntropy;
}

function computeActorLinkStats(graph) {
  const linkStrengthsByActorPair = {};
  const possibleLinkCountsByActorPair = {};
  graph.copyCount = 0;
  for (let i = 0; i < graph.moves.length - 1; i++) {
    const actorA = graph.moves[i]?.actor || 0;
    for (let j = i + 1; j < graph.moves.length; j++) {
      // skip ~verbatim copies
      if ((graph.links[j]?.[i] || 0) >= 0.99) {
        graph.copyCount++;
        continue;
      }
      const actorB = graph.moves[j]?.actor || 0;
      // track the raw link weights of backlinks from actor B to actor A
      const pair = [actorB, actorA].join(":");
      if (!linkStrengthsByActorPair[pair]) {
        linkStrengthsByActorPair[pair] = [];
      }
      linkStrengthsByActorPair[pair].push(graph.links[j]?.[i] || 0);
      // increment count of possible backlinks from actor B to actor A
      if (!possibleLinkCountsByActorPair[pair]) {
        possibleLinkCountsByActorPair[pair] = 0;
      }
      possibleLinkCountsByActorPair[pair] += 1;
    }
  }
  const linkDensitiesByActorPair = {};
  for (const [pair, strengths] of Object.entries(linkStrengthsByActorPair)) {
    linkDensitiesByActorPair[pair] =
      totalLinkWeight(strengths) / possibleLinkCountsByActorPair[pair];
  }
  graph.linkDensitiesByActorPair = linkDensitiesByActorPair;
}

/// data processing

async function embed(str) {
  return (
    await extractor(str, {
      convert_to_tensor: "True",
      pooling: "mean",
      normalize: true,
    })
  )[0];
}

async function computeLinks(moves) {
  // calculate embeddings for each move
  for (const move of moves) {
    // move.embedding = (await embed(move.text)).tolist();
    move.embedding = await embed(move.text);
  }
  // build links table
  const links = {};
  for (let i = 0; i < moves.length; i++) {
    const currMove = moves[i];
    links[i] = {};
    for (let j = 0; j < i; j++) {
      const prevMove = moves[j];
      links[i][j] = cosineSimilarity(currMove.embedding, prevMove.embedding);
    }
  }
  return links;
}

/// react rendering

// Given the rendered locations of two design moves, return the location of the
// right-angle "elbow" between them.
function elbow(pt1, pt2) {
  const x = (pt1.x + pt2.x) / 2;
  const y = pt1.y - (pt2.x - pt1.x) / 2;
  return { x, y };
}

// Given `props` augmented with the `idx` of a design move, return the location
// at which this move should be rendered.
function moveLoc(props) {
  return { x: props.idx * props.moveSpacing + INIT_X, y: INIT_Y };
}

function shouldSegmentTimeline(currMove, prevMove) {
  if (!(currMove?.timestamp && prevMove?.timestamp)) return false;
  const deltaTime =
    Date.parse(currMove.timestamp) - Date.parse(prevMove.timestamp);
  return deltaTime >= SEGMENT_THRESHOLD;
}

function parseCSV(text, maxRows = 10) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return { headers: [], rows: [] };
  const split = (s) => s.split(","); // simple split; fine for your generated CSVs
  const headers = split(lines[0]);
  const rows = lines.slice(1, 1 + maxRows).map((ln) => split(ln));
  return { headers, rows };
}

function ArtifactsPanel({ datasetPath }) {
  const config = ARTIFACTS_BY_DATASET[datasetPath];
  const [csvPreviews, setCsvPreviews] = React.useState([]);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!config?.csvs?.length) {
        setCsvPreviews([]);
        return;
      }
      const previews = [];
      for (const url of config.csvs) {
        try {
          const txt = await (await fetch(url)).text();
          const table = parseCSV(txt, 10);
          previews.push({ url, ...table });
        } catch (e) {
          previews.push({ url, error: true, headers: [], rows: [] });
          console.error("Failed to load CSV", url, e);
        }
      }
      if (!cancelled) setCsvPreviews(previews);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [datasetPath]);

  if (!config) return null;

  return e(
    "aside",
    {
      style: {
        borderLeft: "1px solid #eee",
        paddingLeft: 16,
        paddingTop: 4,
        position: "sticky",
        top: 0,
        alignSelf: "start",
        height: "100%",
        overflow: "auto",
        maxHeight: "calc(100vh - 24px)",
        width: "100%",
      },
    },
    e("h3", { style: { marginTop: 0 } }, "Artifacts"),
    // images
    config.images?.length
      ? e(
          "div",
          {
            style: {
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 20,
              width: "100%",
              overflowX: "auto",
            },
          },
          ...config.images.map((src) =>
            e(
              "a",
              {
                key: src,
                href: src,
                target: "_blank",
                rel: "noreferrer",
                style: {
                  display: "block",
                  maxWidth: "100%",
                  width: "min(100%, 1000px)",
                },
              },
              e("img", {
                src,
                alt: src.split("/").pop(),
                style: {
                  display: "block",
                  width: "100%",
                  height: "auto",
                  objectFit: "contain",
                  borderRadius: 8,
                  boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
                },
              })
            )
          )
        )
      : null,

    // CSV previews
    csvPreviews.map((csv) =>
      e(
        "div",
        {
          key: csv.url,
          style: {
            marginBottom: 16,
            border: "1px solid #eee",
            borderRadius: 8,
            overflow: "hidden",
          },
        },
        e(
          "div",
          {
            style: {
              padding: "8px 10px",
              background: "#fafafa",
              borderBottom: "1px solid #eee",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: 13,
            },
          },
          e("span", null, csv.url.split("/").pop()),
          e(
            "a",
            {
              href: csv.url,
              download: true,
              style: { textDecoration: "none" },
            },
            "Download"
          )
        ),
        csv.error
          ? e(
              "div",
              { style: { padding: 10, color: "crimson", fontSize: 12 } },
              "Could not load preview."
            )
          : e(
              "div",
              { style: { overflowX: "auto" } },
              e(
                "table",
                {
                  style: {
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: 12,
                  },
                },
                e(
                  "thead",
                  null,
                  e(
                    "tr",
                    null,
                    ...csv.headers.map((h, i) =>
                      e(
                        "th",
                        {
                          key: i,
                          style: {
                            textAlign: "left",
                            padding: "6px 8px",
                            borderBottom: "1px solid #eee",
                            whiteSpace: "nowrap",
                          },
                        },
                        h
                      )
                    )
                  )
                ),
                e(
                  "tbody",
                  null,
                  ...csv.rows.map((row, r) =>
                    e(
                      "tr",
                      { key: r },
                      ...row.map((cell, c) =>
                        e(
                          "td",
                          {
                            key: c,
                            style: {
                              padding: "6px 8px",
                              borderBottom: "1px solid #f5f5f5",
                              whiteSpace: "nowrap",
                            },
                          },
                          cell
                        )
                      )
                    )
                  )
                )
              )
            )
      )
    )
  );
}

function DesignMove(props) {
  const move = props.moves[props.idx];
  const currLoc = moveLoc(props);
  const scaledForelinkWeight = scale(
    move.forelinkWeight,
    [0, props.maxForelinkWeight],
    [0, MOVE_LINK_BAR_HEIGHT]
  );
  const scaledBacklinkWeight = scale(
    move.backlinkWeight,
    [0, props.maxBacklinkWeight],
    [0, MOVE_LINK_BAR_HEIGHT]
  );
  const moveLinkBarSize = 10 + MOVE_LINK_BAR_HEIGHT + 10;
  // create correct move marker based on actor
  let moveMarker = null;
  const actorId = move.actor || 0;
  const fillColor = rgbToStr(colorForActor(actorId));

  if (actorId === 0) {
    // circle
    moveMarker = e("circle", {
      cx: currLoc.x,
      cy: currLoc.y,
      r: 5,
      fill: fillColor,
    });
  } else if (actorId === 1) {
    // square
    moveMarker = e("rect", {
      x: currLoc.x - 5,
      y: currLoc.y - 5,
      width: 10,
      height: 10,
      fill: fillColor,
    });
  } else if (actorId === 2) {
    // triangle (pointing up)
    const p = [
      [currLoc.x, currLoc.y - 6],
      [currLoc.x - 6, currLoc.y + 5],
      [currLoc.x + 6, currLoc.y + 5],
    ]
      .map(([x, y]) => `${x},${y}`)
      .join(" ");
    moveMarker = e("polygon", { points: p, fill: fillColor });
  } else {
    // fallback: small diamond
    const p = [
      [currLoc.x, currLoc.y - 6],
      [currLoc.x - 6, currLoc.y],
      [currLoc.x, currLoc.y + 6],
      [currLoc.x + 6, currLoc.y],
    ]
      .map(([x, y]) => `${x},${y}`)
      .join(" ");
    moveMarker = e("polygon", { points: p, fill: fillColor });
  }
  return e(
    "g",
    {},
    MOVE_TEXT_MODE === "FULL"
      ? e(
          "text",
          {
            x: currLoc.x + 5,
            y: currLoc.y - moveLinkBarSize,
            transform: `rotate(270, ${currLoc.x + 5}, ${
              currLoc.y - moveLinkBarSize
            })`,
            fontWeight:
              move.backlinkCriticalMove || move.forelinkCriticalMove
                ? "bold"
                : "normal",
          },
          move.text
        )
      : null,
    MOVE_TEXT_MODE === "INDEX"
      ? e(
          "text",
          {
            x: currLoc.x,
            y: currLoc.y - moveLinkBarSize,
            textAnchor: "middle",
            fontWeight:
              move.backlinkCriticalMove || move.forelinkCriticalMove
                ? "bold"
                : "normal",
          },
          props.idx
        )
      : null,
    e("rect", {
      x: currLoc.x - 5,
      y: currLoc.y - 10 - scaledBacklinkWeight,
      width: 5,
      height: scaledBacklinkWeight,
      fill: "#998ec3",
    }),
    e("rect", {
      x: currLoc.x,
      y: currLoc.y - 10 - scaledForelinkWeight,
      width: 5,
      height: scaledForelinkWeight,
      fill: "#f1a340",
    }),
    moveMarker
  );
}

function makeTimelineDividers(props) {
  const dividers = [];
  for (let idx = 0; idx < props.moves.length; idx++) {
    const currLoc = moveLoc({ ...props, idx });
    const splitAfter = shouldSegmentTimeline(
      props.moves[idx + 1],
      props.moves[idx]
    );
    if (!splitAfter) continue;
    dividers.push(
      e("line", {
        stroke: "#999",
        strokeDasharray: "2",
        strokeWidth: 1,
        x1: currLoc.x + props.moveSpacing / 2,
        y1: currLoc.y - GRAPH_WIDTH,
        x2: currLoc.x + props.moveSpacing / 2,
        y2: currLoc.y + GRAPH_WIDTH,
      })
    );
  }
  return dividers;
}

function makeLinkObjects(props) {
  const linkLines = [];
  const linkJoints = [];
  for (const [currIdx, linkSet] of Object.entries(props.links)) {
    const currLoc = moveLoc({ ...props, idx: currIdx });
    for (const [prevIdx, strength] of Object.entries(linkSet)) {
      if (strength < MIN_LINK_STRENGTH) continue; // skip weak connections (arbitrary threshold)
      const prevLoc = moveLoc({ ...props, idx: prevIdx });
      const jointLoc = elbow(currLoc, prevLoc);
      const lineStrength = scale(strength, [MIN_LINK_STRENGTH, 1], [255, 0]);
      let color = "";
      if (props.actors.size > 1 && SHOULD_COLORIZE_LINKS) {
        const currActor = props.moves[currIdx].actor || 0;
        const prevActor = props.moves[prevIdx].actor || 0;
        let targetColor = null;
        if (currActor === prevActor && currActor === 0) {
          targetColor = { red: 255, green: 0, blue: 0 };
        } else if (currActor === prevActor && currActor === 1) {
          targetColor = { red: 0, green: 0, blue: 255 };
        } else {
          targetColor = { red: 160, green: 0, blue: 255 };
        }
        const r = scale(
          strength,
          [MIN_LINK_STRENGTH, 1],
          [255, targetColor.red]
        );
        const g = scale(
          strength,
          [MIN_LINK_STRENGTH, 1],
          [255, targetColor.green]
        );
        const b = scale(
          strength,
          [MIN_LINK_STRENGTH, 1],
          [255, targetColor.blue]
        );
        color = `rgb(${r},${g},${b})`;
      } else {
        color = `rgb(${lineStrength},${lineStrength},${lineStrength})`;
      }
      linkLines.push({
        x1: currLoc.x,
        y1: currLoc.y,
        x2: jointLoc.x,
        y2: jointLoc.y,
        color,
        strength,
      });
      linkLines.push({
        x1: prevLoc.x,
        y1: prevLoc.y,
        x2: jointLoc.x,
        y2: jointLoc.y,
        color,
        strength,
      });
      linkJoints.push({ x: jointLoc.x, y: jointLoc.y, color, strength });
    }
  }
  return { linkLines, linkJoints };
}

function FuzzyLinkograph(props) {
  const dividers = makeTimelineDividers(props);
  const { linkLines, linkJoints } = makeLinkObjects(props);
  return e(
    "div",
    { className: "fuzzy-linkograph" },
    e("h2", {}, props.title),
    // e(
    //   "svg",
    //   { viewBox: `0 0 ${GRAPH_WIDTH} ${GRAPH_WIDTH / 2 + INIT_Y}` },
    e(
      "svg",
      {
        viewBox: `0 0 ${GRAPH_WIDTH} ${GRAPH_WIDTH / 2 + INIT_Y}`,
        style: { width: "100%", height: "auto", display: "block" },
        preserveAspectRatio: "xMidYMid meet",
      },
      ...dividers,
      ...linkLines
        .sort((a, b) => a.strength - b.strength)
        .map((line) => {
          return e("line", {
            x1: line.x1,
            y1: line.y1,
            x2: line.x2,
            y2: line.y2,
            stroke: line.color,
            strokeWidth: 2,
          });
        }),
      ...linkJoints
        .sort((a, b) => a.strength - b.strength)
        .map((joint) => {
          return e("circle", {
            cx: joint.x,
            cy: joint.y,
            r: 3,
            fill: joint.color,
          });
        }),
      ...props.moves.map((_, idx) => e(DesignMove, { ...props, idx }))
    )
  );
}

const DATASET_OPTIONS = [
  {
    label: "Baseline Roadtrip Session",
    path: "./data/teamx_session_10min_linked.json",
    desc: "Original 10-minute collaborative session with realistic pacing and natural idea flow (baseline). Uses sentence-transformers embedding. Model = all-MiniLM-L6-v2.",
  },
  {
    label: "Multi-Context Planning (V1)",
    path: "./data/teamx_V1_linked.json",
    desc: "Expanded topical diversity (travel, event, gadget). Includes verbs like add_note and link for structural clarity. Uses sentence-transformers embedding. Model = all-MiniLM-L6-v2.",
  },
  {
    label: "Detour-Rich Flow (V2)",
    path: "./data/teamx_v2_linked.json",
    desc: "Introduces off-topic additions and early integrations to simulate irregular collaboration and bumpier link curves. Uses CLIP embedding",
  },
  {
    label: "Naturalized Content Flow (V3)",
    path: "./data/teamx_v3_linked.json",
    desc: "Cleans action verbs from text for natural phrasing; focuses on semantic content for CLIP embedding analysis. Uses CLIP embedding.",
  },
  {
    label: "RoadTrip to Colorado, Utah, and Vegas",
    path: "./data/team_roadtrip_session_10min_linked.json",
    desc: "Roadtrip plan. Uses CLIP embedding.",
  },
];

// 2) Simple cache so we only fetch/compute once per dataset.
const EPISODE_CACHE = new Map();

/**
 * Normalizes a loaded JSON file to a single "episode" object with:
 *  - title, moves, links, actors, moveSpacing, stats computed
 */
async function normalizeAndComputeEpisode(jsonLike, fallbackTitle = "Episode") {
  // Some files are { id: { title, moves, links? } }, others may be just { moves, links? }.
  let firstKey = null;
  if (!Array.isArray(jsonLike)) {
    const keys = Object.keys(jsonLike);
    if (
      keys.length === 1 &&
      (jsonLike[keys[0]].moves || Array.isArray(jsonLike[keys[0]]))
    ) {
      firstKey = keys[0];
      jsonLike = jsonLike[firstKey];
    }
  }

  const title = jsonLike.title || firstKey || fallbackTitle;
  const moves = jsonLike.moves || jsonLike; // if no .moves, assume the whole object is the moves array
  let links = jsonLike.links;

  // If no links, compute them (embeddings are added to moves).
  if (!links) {
    links = await computeLinks(moves);
  }

  // Prepare the episode and compute all stats
  const episode = {
    title,
    moves,
    links,
  };

  episode.actors = new Set(episode.moves.map((m) => m.actor || 0));
  computeLinkDensityIndex(episode);
  computeMoveWeights(episode);
  computeEntropy(episode);
  if (episode.actors.size > 1) {
    computeActorLinkStats(episode);
  }
  episode.moveSpacing =
    (GRAPH_WIDTH - INIT_X * 2) / Math.max(1, episode.moves.length - 1);
  return episode;
}

async function loadEpisode(path) {
  // Return from cache if present
  if (EPISODE_CACHE.has(path)) return EPISODE_CACHE.get(path);

  // Fetch and process
  let episode = null;
  try {
    const res = await fetch(path);
    const json = await res.json();
    episode = await normalizeAndComputeEpisode(json, path.split("/").pop());
  } catch (err) {
    console.error("Failed to load dataset:", path, err);
    throw err;
  }

  EPISODE_CACHE.set(path, episode);
  return episode;
}

// 3) Controls + Single Graph App

// function HeaderControls({ datasetPath, onChange, episode }) {
//   return e(
//     "div",
//     {
//       style: {
//         display: "flex",
//         gap: "12px",
//         alignItems: "center",
//         marginBottom: "12px",
//       },
//     },
//     e(
//       "label",
//       { htmlFor: "datasetSelect", style: { fontWeight: "600" } },
//       "Dataset"
//     ),
//     e(
//       "select",
//       {
//         id: "datasetSelect",
//         value: datasetPath,
//         onChange: (evt) => onChange(evt.target.value),
//         style: { padding: "6px 8px" },
//       },
//       DATASET_OPTIONS.map((opt) =>
//         e("option", { key: opt.path, value: opt.path }, opt.label)
//       )
//     ),
//     episode
//       ? e(
//           "div",
//           { style: { marginLeft: "auto", opacity: 0.8 } },
//           `Moves: ${episode.moves.length} • Actors: ${episode.actors.size}`
//         )
//       : null
//   );
// }

function HeaderControls({ datasetPath, onChange, episode }) {
  const selected =
    DATASET_OPTIONS.find((d) => d.path === datasetPath) || DATASET_OPTIONS[0];

  return e(
    "div",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        marginBottom: "12px",
      },
    },
    // Row: label + select + quick counts (aligned right)
    e(
      "div",
      { style: { display: "flex", gap: "12px", alignItems: "center" } },
      e(
        "label",
        { htmlFor: "datasetSelect", style: { fontWeight: 600 } },
        "Dataset"
      ),
      e(
        "select",
        {
          id: "datasetSelect",
          value: datasetPath,
          onChange: (evt) => onChange(evt.target.value),
          style: { padding: "6px 8px" },
        },
        DATASET_OPTIONS.map((opt) =>
          e("option", { key: opt.path, value: opt.path }, opt.label)
        )
      ),
      episode
        ? e(
            "div",
            {
              style: {
                marginLeft: "auto",
                opacity: 0.8,
                fontSize: 13,
                whiteSpace: "nowrap",
              },
            },
            `Moves: ${episode.moves.length} • Actors: ${episode.actors.size}`
          )
        : null
    ),

    // Description block (updates when selection changes)
    e(
      "div",
      {
        id: "dataset-description",
        style: {
          fontSize: "0.92em",
          color: "#555",
          padding: "6px 8px",
          background: "#fafafa",
          border: "1px solid #eee",
          borderRadius: 8,
          lineHeight: 1.35,
          maxWidth: 800,
        },
      },
      selected?.desc || ""
    )
  );
}

// function StatsBar({ episode }) {
//   if (!episode) return null;

//   const {
//     linkDensityIndex,
//     entropy,
//     backlinkEntropy,
//     forelinkEntropy,
//     horizonlinkEntropy,
//     copyCount,
//     maxForelinkWeight,
//     maxBacklinkWeight,
//   } = episode;

//   const wrap = (label, value) =>
//     e(
//       "div",
//       { style: { display: "flex", gap: 6, alignItems: "baseline" } },
//       e("div", { style: { fontSize: 12, opacity: 0.7 } }, label),
//       e("div", { style: { fontWeight: 600 } }, value)
//     );

//   const minutes = (ms) => (ms / 60000).toFixed(0);

//   return e(
//     "div",
//     {
//       style: {
//         display: "grid",
//         gridTemplateColumns: "repeat(8, minmax(0,1fr))", // widened to fit new fields
//         gap: 12,
//         padding: "8px 0 16px 0",
//         borderBottom: "1px solid #eee",
//         marginBottom: 12,
//       },
//     },
//     // row 1 (core metrics)
//     wrap("Link density", linkDensityIndex?.toFixed(3)),
//     wrap("Entropy (total)", entropy?.toFixed(3)),
//     wrap("Entropy back", backlinkEntropy?.toFixed(3)),
//     wrap("Entropy fore", forelinkEntropy?.toFixed(3)),
//     wrap("Entropy horizon", horizonlinkEntropy?.toFixed(3)),
//     wrap("Near-copies", copyCount ?? 0),
//     wrap("Max fore W", maxForelinkWeight?.toFixed(2)),
//     wrap("Max back W", maxBacklinkWeight?.toFixed(2)),

//     // row 2 (global settings / thresholds)
//     wrap("Sim threshold", MIN_LINK_STRENGTH.toFixed(2)),
//     wrap("Segment gap", `${minutes(SEGMENT_THRESHOLD)} min`),
//     wrap("Colorized links", SHOULD_COLORIZE_LINKS ? "on" : "off"),
//     wrap("Text mode", MOVE_TEXT_MODE)
//   );
// }

function StatsBar({ episode }) {
  if (!episode) return null;

  const {
    linkDensityIndex,
    entropy,
    backlinkEntropy,
    forelinkEntropy,
    horizonlinkEntropy,
    copyCount,
    maxForelinkWeight,
    maxBacklinkWeight,
  } = episode;

  // Wrap a label/value pair with a tooltip (native title attr)
  const wrap = (label, value) =>
    e(
      "div",
      {
        title: METRIC_INFO[label] || "", // ← hover tooltip
        style: {
          display: "flex",
          gap: 6,
          alignItems: "baseline",
          cursor: "help", // visual hint that it's explainable
        },
      },
      e("div", { style: { fontSize: 12, opacity: 0.7 } }, label),
      e("div", { style: { fontWeight: 600 } }, value)
    );

  const minutes = (ms) => (ms / 60000).toFixed(0);

  return e(
    "div",
    {
      style: {
        display: "grid",
        gridTemplateColumns: "repeat(8, minmax(0,1fr))",
        gap: 12,
        padding: "8px 0 16px 0",
        borderBottom: "1px solid #eee",
        marginBottom: 12,
      },
    },
    // row 1 (core metrics)
    wrap("Link density", linkDensityIndex?.toFixed(3)),
    wrap("Entropy (total)", entropy?.toFixed(3)),
    wrap("Entropy back", backlinkEntropy?.toFixed(3)),
    wrap("Entropy fore", forelinkEntropy?.toFixed(3)),
    wrap("Entropy horizon", horizonlinkEntropy?.toFixed(3)),
    wrap("Near-copies", copyCount ?? 0),
    wrap("Max fore W", maxForelinkWeight?.toFixed(2)),
    wrap("Max back W", maxBacklinkWeight?.toFixed(2)),

    // row 2 (global settings / thresholds)
    wrap("Sim threshold", MIN_LINK_STRENGTH.toFixed(2)),
    wrap("Segment gap", `${minutes(SEGMENT_THRESHOLD)} min`),
    wrap("Colorized links", SHOULD_COLORIZE_LINKS ? "on" : "off"),
    wrap("Text mode", MOVE_TEXT_MODE)
  );
}

function App() {
  const [datasetPath, setDatasetPath] = React.useState(DATASET_OPTIONS[0].path);
  const [episode, setEpisode] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  const load = React.useCallback(async (path) => {
    try {
      setLoading(true);
      setError(null);
      const ep = await loadEpisode(path);
      setEpisode(ep);
    } catch (err) {
      setError("Could not load dataset. See console for details.");
      setEpisode(null);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    load(datasetPath);
  }, [datasetPath, load]);

  return e(
    "div",
    {
      style: {
        width: "100%",
        maxWidth: "unset",
        margin: 0,
        padding: "12px 16px",
      },
    },

    e(HeaderControls, {
      datasetPath,
      onChange: setDatasetPath,
      episode,
    }),
    error
      ? e("div", { style: { color: "crimson", marginBottom: 12 } }, error)
      : null,
    e(
      "div",
      {
        style: {
          display: "grid",
          // gridTemplateColumns: ARTIFACTS_BY_DATASET[datasetPath]
          //   ? "1fr 320px"
          //   : "1fr",
          // gridTemplateColumns: ARTIFACTS_BY_DATASET[datasetPath]
          //   ? "minmax(0,1fr) 320px"
          //   : "minmax(0,1fr)",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          alignItems: "start",
        },
      },
      // LEFT: graph + stats
      loading
        ? e("div", { style: { opacity: 0.75 } }, "Loading and computing…")
        : e(
            "div",
            { style: { minWidth: 0 } },
            e(StatsBar, { episode }),
            episode
              ? e(FuzzyLinkograph, {
                  title: episode.title,
                  moves: episode.moves,
                  links: episode.links,
                  actors: episode.actors,
                  moveSpacing: episode.moveSpacing,
                  maxForelinkWeight: episode.maxForelinkWeight,
                  maxBacklinkWeight: episode.maxBacklinkWeight,
                })
              : null
          ),
      // RIGHT: artifacts (only if configured)
      // e(ArtifactsPanel, { datasetPath })
      e(ArtifactsPanel, { datasetPath, style: { minWidth: 0 } })
    )
  );
}

// 4) Mount once; the App handles dataset switching.
let root = null;
function renderUI() {
  if (!root) {
    root = ReactDOM.createRoot(document.getElementById("app"));
  }
  root.render(e(App));
}

// 5) Boot
renderUI();

// main();
