import flask
server = flask.Flask(__name__)
import sys
import numpy as np
from urllib import request

import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

blacklist = []

@server.route('/')
def main():
  class args:
    input            = flask.request.args.get('input', type=str, default=None)
    model_name       = flask.request.args.get('model_name', type=str, default='RealESRGAN_x4plus',)
    denoise_strength = flask.request.args.get('denoise_strength', type=float, default=0.5)
    outscale         = flask.request.args.get('outscale', type=float, default=4)
    model_path       = flask.request.args.get('model_path', type=str, default=None)
    suffix           = flask.request.args.get('suffix', type=str, default='out')
    tile             = flask.request.args.get('tile', type=int, default=0)
    tile_pad         = flask.request.args.get('tile_pad', type=int, default=10)
    pre_pad          = flask.request.args.get('pre_pad', type=int, default=0)
    face_enhance     = flask.request.args.get('face_enhance', type=bool, default=False)
    fp32             = flask.request.args.get('fp32', type=bool, default=False)
    alpha_upsampler  = flask.request.args.get('alpha_upsampler', type=str, default='realesrgan')
    ext              = flask.request.args.get('ext', type=str, default='auto')
    gpu_id           = flask.request.args.get('gpu-id', type=int, default=None)

  bot_id = flask.request.args.get('bot_id', type=int)
  user_id = flask.request.args.get('user_id', type=int)
  print('BotID：', bot_id, '\n用户ID：', user_id, '\n图片修复：', args.input, sep='')
  if not user_id or user_id in blacklist:
    return '无效请求'

  # determine models according to model names
  args.model_name = args.model_name.split('.')[0]
  if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
  elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
  elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
  elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    file_url = ['https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
  elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    netscale = 4
    file_url = ['https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
  elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    netscale = 4
    file_url = [
      'https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
      'https://ghproxy.com/github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    ]

  # determine model paths
  if args.model_path is not None:
    model_path = args.model_path
  else:
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
      ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
      for url in file_url:
        # model_path will be updated
        model_path = load_file_from_url(
          url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

  # use dni to control the denoise strength
  dni_weight = None
  if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
    wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    model_path = [model_path, wdn_model_path]
    dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

  # restorer
  upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=args.tile,
    tile_pad=args.tile_pad,
    pre_pad=args.pre_pad,
    half=not args.fp32,
    gpu_id=args.gpu_id)

  if args.face_enhance:  # Use GFPGAN for face enhancement
    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
      model_path='https://ghproxy.com/github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
      upscale=args.outscale,
      arch='clean',
      channel_multiplier=2,
      bg_upsampler=upsampler)

  try:
    res = request.urlopen(args.input)
    img = np.asarray(bytearray(res.read()), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
      img_mode = 'RGBA'
    else:
      img_mode = None

    if args.face_enhance:
      _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
      output, _ = upsampler.enhance(img, outscale=args.outscale)

    bytes = cv2.imencode('.jpg', output)[1].tobytes()
  except Exception as e:
    print('错误：', e, sep='')
    return flask.redirect('http://ovooa.com/API/yi')
  return flask.Response(bytes, mimetype='image/jpeg')

if __name__ == '__main__':
  server.run(host='0.0.0.0', port=sys.argv[1])