
import io
from typing import Tuple

from PIL import Image, ExifTags, UnidentifiedImageError

from api.config import get_settings

settings = get_settings()

ALLOWED_MIME_MAGIC: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
}
ANIMATED_FORMATS = {"gif", "webp", "apng"}


class ImageValidationError(Exception):
    
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class ImageValidationService:
   

    def validate_and_open(self, file_bytes: bytes, filename: str = "") -> Image.Image:
        
        self._check_size(file_bytes)
        self._check_magic_bytes(file_bytes, filename)
        image = self._open_image(file_bytes)
        self._check_animated(image, filename)
        image = self._fix_orientation(image)
        image = self._convert_to_rgb(image)
        self._check_dimensions(image)
        return image

   

    def _check_size(self, data: bytes) -> None:
        size = len(data)
        if size == 0:
            raise ImageValidationError("File is empty.", status_code=413)
        if size > settings.max_upload_bytes:
            raise ImageValidationError(
                f"File size {size / (1024*1024):.1f} MB exceeds the "
                f"{settings.max_upload_mb} MB limit.",
                status_code=413,
            )

    def _check_magic_bytes(self, data: bytes, filename: str) -> None:
        
        for magic, _ in ALLOWED_MIME_MAGIC.items():
            if data[:len(magic)] == magic:
                return
        raise ImageValidationError(
            f"'{filename}' is not a supported image format. "
            "Accepted formats: JPEG, PNG, BMP.",
            status_code=422,
        )

    def _open_image(self, data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()                          
            img = Image.open(io.BytesIO(data))   
            return img
        except UnidentifiedImageError:
            raise ImageValidationError("Cannot identify image format.", status_code=422)
        except Exception:
            raise ImageValidationError("Image file is corrupted or cannot be opened.", status_code=422)

    def _check_animated(self, image: Image.Image, filename: str) -> None:
        fmt = (image.format or "").lower()
        if fmt in ANIMATED_FORMATS:
            raise ImageValidationError(
                f"Animated images are not supported (detected: {fmt.upper()}).",
                status_code=422,
            )
       
        try:
            image.seek(1)
            raise ImageValidationError("Multi-frame / animated images are not supported.", status_code=422)
        except EOFError:
            pass 

    def _fix_orientation(self, image: Image.Image) -> Image.Image:
        
        try:
            exif = image._getexif()
            if exif is None:
                return image
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                rotations = {3: 180, 6: 270, 8: 90}
                if orientation in rotations:
                    image = image.rotate(rotations[orientation], expand=True)
        except Exception:
            pass  
        return image

    def _convert_to_rgb(self, image: Image.Image) -> Image.Image:
       
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _check_dimensions(self, image: Image.Image) -> None:
        w, h = image.size
        min_d = settings.min_image_dimension
        max_d = settings.max_image_dimension

        if w < min_d or h < min_d:
            raise ImageValidationError(
                f"Image dimensions {w}×{h} are too small. Minimum: {min_d}×{min_d}.",
                status_code=422,
            )
        if w > max_d or h > max_d:
            
            pass  
