import csv
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VideoMetadata:
    video_id: str
    published_at: Optional[datetime]
    channel_id: str
    title: str
    thumbnail_maxres_url: str
    category_id: str
    default_language: str
    duration: str
    view_count: int
    like_count: int
    favorite_count: int
    comment_count: int

    @classmethod
    def from_api(cls, item: Dict) -> "VideoMetadata":
        """
        Create a VideoMetadata instance from a raw YouTube API video resource.
        """
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        stats = item.get("statistics", {})

        # Parse publishedAt into a datetime
        pub_str = snippet.get("publishedAt", "")
        published_at = None
        if pub_str:
            published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))

        # Thumbnail URL (max resolution)
        thumbnails = snippet.get("thumbnails", {})
        maxres = thumbnails.get("maxres") or {}
        thumbnail_url = maxres.get("url", "")

        return cls(
            video_id=item.get("id", ""),
            published_at=published_at,
            channel_id=snippet.get("channelId", ""),
            title=snippet.get("title", ""),
            thumbnail_maxres_url=thumbnail_url,
            category_id=snippet.get("categoryId", ""),
            default_language=snippet.get("defaultLanguage", ""),
            duration=content_details.get("duration", ""),
            view_count=int(stats.get("viewCount", 0)),
            like_count=int(stats.get("likeCount", 0)),
            favorite_count=int(stats.get("favoriteCount", 0)),
            comment_count=int(stats.get("commentCount", 0)),
        )


    @classmethod
    def from_dict(cls, row: Dict[str, str]) -> "VideoMetadata":
        """
        Create an instance from a CSV row (dict of strings, as returned by csv.DictReader).
        """
        # parse published_at
        pub = row.get("published_at", "")
        published_at = datetime.fromisoformat(pub) if pub else None

        return cls(
            video_id           = row["video_id"],
            published_at       = published_at,
            channel_id         = row["channel_id"],
            title              = row["title"],
            thumbnail_maxres_url=row["thumbnail_maxres_url"],
            category_id        = row["category_id"],
            default_language   = row["default_language"],
            duration           = row["duration"],
            view_count         = int(row["view_count"]),
            like_count         = int(row["like_count"]),
            favorite_count     = int(row["favorite_count"]),
            comment_count      = int(row["comment_count"]),
        )


class ChannelVideos:
    """
    Collects VideoMetadata for a single channel and maintains
    total_videos, total_views, and average_view_count as you add items.
    """
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self.videos: List[VideoMetadata] = []
        self.total_videos: int = 0
        self.total_views: int = 0
        self.average_view_count: float = 0.0

    def add(self, video: VideoMetadata) -> None:
        if video.channel_id != self.channel_id:
            raise ValueError(
                f"Cannot add video from channel {video.channel_id} "
                f"to ChannelVideos({self.channel_id})"
            )
        self.videos.append(video)
        self.total_videos += 1
        self.total_views += video.view_count
        self.average_view_count = self.total_views / self.total_videos

    def add_all(self, videos: List[VideoMetadata]) -> None:
        for v in videos:
            self.add(v)

    def export_to_csv(self, csv_path: str) -> None:
        """
        Export the collected video metadata to a CSV file.

        Args:
            csv_path: Path to the output CSV file.
        """

        # Define CSV column headers in order
        fieldnames = [
            'video_id', 'published_at', 'channel_id', 'title',
            'thumbnail_maxres_url', 'category_id', 'default_language',
            'duration', 'view_count', 'like_count',
            'favorite_count', 'comment_count'
        ]

        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for video in self.videos:
                row = asdict(video)
                # Convert datetime to ISO format string
                if video.published_at:
                    row['published_at'] = video.published_at.isoformat()
                else:
                    row['published_at'] = ''
                writer.writerow(row)

    @classmethod
    def from_csv(cls, channel_id: str, csv_path: str) -> "ChannelVideos":
        """
        Factory: read each CSV row via VideoMetadata.from_dict,
        filter by channel_id, and build the ChannelVideos.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {csv_path}")

        cv = cls(channel_id)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video = VideoMetadata.from_dict(row)
                if video.channel_id == channel_id:
                    cv.add(video)
        return cv
    

class ChannelVideosCollection:
    """
    Load a CSV once and partition rows into ChannelVideos buckets.
    """

    def __init__(self):
        # channel_id → ChannelVideos
        self.by_channel: Dict[str, ChannelVideos] = {}

    @classmethod
    def load_all(cls, csv_path: str) -> "ChannelVideosCollection":
        coll = cls()
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {csv_path}")

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = VideoMetadata.from_dict(row)
                cid = vid.channel_id

                bucket = coll.by_channel.get(cid)
                if bucket is None:
                    bucket = ChannelVideos(cid)
                    coll.by_channel[cid] = bucket

                bucket.add(vid)

        return coll

    def get(self, channel_id: str) -> ChannelVideos:
        return self.by_channel[channel_id]
