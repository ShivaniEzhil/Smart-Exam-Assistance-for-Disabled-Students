"""
Download a targeted subset of WLASL videos for exam-relevant words.
Downloads non-YouTube videos via urllib and YouTube via yt-dlp.

Usage (from accessible-exam):
  python sign_language_training/download_wlasl_subset.py [--max-glosses 20] [--max-per-gloss 10]
"""
import argparse
import json
import os
import sys
import time
import random
import urllib.request
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WLASL_JSON = os.path.join(SCRIPT_DIR, "WLASL-master", "start_kit", "WLASL_v0.3.json")
DEFAULT_SAVE = os.path.join(SCRIPT_DIR, "WLASL-master", "start_kit", "videos")

# Useful words for exam context — ordered by priority
EXAM_WORDS = [
    "book", "help", "yes", "no", "computer", "go", "drink", "finish",
    "write", "read", "school", "teacher", "student", "question", "answer",
    "think", "know", "understand", "wrong", "right", "please", "thank you",
    "sorry", "time", "work", "study", "learn", "test", "paper", "word",
    "name", "number", "water", "good", "bad", "more", "all", "stop",
    "start", "again", "what", "who", "where", "when", "how", "why",
    "before", "after", "now", "wait", "chair", "table", "mother", "father",
    "deaf", "need", "want", "like", "can", "sit",
]


def download_direct(url, saveto, referer=""):
    """Download a non-YouTube video via urllib."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, None, headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        with open(saveto, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    FAIL (direct): {e}")
        return False


def download_youtube(url, saveto):
    """Download a YouTube video using yt-dlp."""
    try:
        cmd = [
            "yt-dlp", "-f", "mp4/best",
            "--no-playlist",
            "-o", saveto,
            "--quiet",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"    FAIL (yt-dlp): {e}")
        return False


def download_video(url, video_id, save_dir):
    """Download one video, returns True on success."""
    # Check if already exists
    mp4_path = os.path.join(save_dir, f"{video_id}.mp4")
    if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 1000:
        return True  # already have it

    if "youtube" in url or "youtu.be" in url:
        return download_youtube(url, mp4_path)
    elif "aslpro" in url:
        # aslpro videos are .swf, skip them
        return False
    else:
        referer = ""
        if "aslbricks" in url:
            referer = "http://aslbricks.org/"
        ok = download_direct(url, mp4_path, referer=referer)
        if ok and os.path.getsize(mp4_path) < 500:
            os.remove(mp4_path)
            return False
        return ok


def main():
    parser = argparse.ArgumentParser(description="Download WLASL videos for exam words")
    parser.add_argument("--max-glosses", type=int, default=20, help="Max glosses to download (default 20)")
    parser.add_argument("--max-per-gloss", type=int, default=8, help="Max videos per gloss (default 8)")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE, help="Directory to save videos")
    args = parser.parse_args()

    with open(WLASL_JSON, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Build gloss -> instances mapping
    gloss_map = {}
    for entry in content:
        gloss_map[entry["gloss"].lower()] = entry

    # Select glosses that exist in WLASL, ordered by EXAM_WORDS priority
    selected = []
    for word in EXAM_WORDS:
        if word in gloss_map and len(selected) < args.max_glosses:
            selected.append(gloss_map[word])
    # If we still have room, fill from WLASL top by instance count
    if len(selected) < args.max_glosses:
        by_count = sorted(content, key=lambda e: -len(e.get("instances", [])))
        for entry in by_count:
            if entry not in selected and len(selected) < args.max_glosses:
                selected.append(entry)

    os.makedirs(args.save_dir, exist_ok=True)
    total_ok, total_fail = 0, 0

    for entry in selected:
        gloss = entry["gloss"]
        instances = entry.get("instances", [])[:args.max_per_gloss]
        print(f"\n[{gloss}] ({len(instances)} clips to download)")

        for inst in instances:
            vid = str(inst["video_id"])
            url = inst["url"]
            ok = download_video(url, vid, args.save_dir)
            if ok:
                total_ok += 1
                print(f"  ✓ {vid}")
            else:
                total_fail += 1
                print(f"  ✗ {vid}")
            time.sleep(random.uniform(0.3, 0.8))

    print(f"\nDone. Downloaded: {total_ok}, Failed: {total_fail}")
    print(f"Videos are in: {args.save_dir}")
    print("Next: python sign_language_training/extract_wlasl_hand_sequences.py")


if __name__ == "__main__":
    main()
