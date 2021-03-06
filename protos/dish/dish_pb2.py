# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/dish/dish.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/dish/dish.proto',
  package='dish',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x16protos/dish/dish.proto\x12\x04\x64ish\"\t\n\x07Request\"\x18\n\x08Response\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x16\n\x05\x41rray\x12\r\n\x05\x61rray\x18\x01 \x03(\x05\"\"\n\x03Img\x12\x1b\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x0b.dish.Array\"\"\n\x07Img_rgb\x12\x17\n\x04imgs\x18\x01 \x03(\x0b\x32\t.dish.Img\")\n\x06\x43ircle\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01r\x18\x03 \x01(\x02\"\'\n\x07\x43ircles\x12\x1c\n\x06\x63ircle\x18\x01 \x03(\x0b\x32\x0c.dish.Circle2^\n\x04\x44ish\x12,\n\nGetCircles\x12\r.dish.Img_rgb\x1a\r.dish.Circles\"\x00\x12(\n\x06GetImg\x12\r.dish.Request\x1a\r.dish.Img_rgb\"\x00\x62\x06proto3')
)




_REQUEST = _descriptor.Descriptor(
  name='Request',
  full_name='dish.Request',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=41,
)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='dish.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='dish.Response.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=43,
  serialized_end=67,
)


_ARRAY = _descriptor.Descriptor(
  name='Array',
  full_name='dish.Array',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='array', full_name='dish.Array.array', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=69,
  serialized_end=91,
)


_IMG = _descriptor.Descriptor(
  name='Img',
  full_name='dish.Img',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='arrays', full_name='dish.Img.arrays', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=127,
)


_IMG_RGB = _descriptor.Descriptor(
  name='Img_rgb',
  full_name='dish.Img_rgb',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='imgs', full_name='dish.Img_rgb.imgs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=129,
  serialized_end=163,
)


_CIRCLE = _descriptor.Descriptor(
  name='Circle',
  full_name='dish.Circle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='dish.Circle.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='dish.Circle.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='r', full_name='dish.Circle.r', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=165,
  serialized_end=206,
)


_CIRCLES = _descriptor.Descriptor(
  name='Circles',
  full_name='dish.Circles',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='circle', full_name='dish.Circles.circle', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=247,
)

_IMG.fields_by_name['arrays'].message_type = _ARRAY
_IMG_RGB.fields_by_name['imgs'].message_type = _IMG
_CIRCLES.fields_by_name['circle'].message_type = _CIRCLE
DESCRIPTOR.message_types_by_name['Request'] = _REQUEST
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['Array'] = _ARRAY
DESCRIPTOR.message_types_by_name['Img'] = _IMG
DESCRIPTOR.message_types_by_name['Img_rgb'] = _IMG_RGB
DESCRIPTOR.message_types_by_name['Circle'] = _CIRCLE
DESCRIPTOR.message_types_by_name['Circles'] = _CIRCLES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Request = _reflection.GeneratedProtocolMessageType('Request', (_message.Message,), dict(
  DESCRIPTOR = _REQUEST,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Request)
  ))
_sym_db.RegisterMessage(Request)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSE,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Response)
  ))
_sym_db.RegisterMessage(Response)

Array = _reflection.GeneratedProtocolMessageType('Array', (_message.Message,), dict(
  DESCRIPTOR = _ARRAY,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Array)
  ))
_sym_db.RegisterMessage(Array)

Img = _reflection.GeneratedProtocolMessageType('Img', (_message.Message,), dict(
  DESCRIPTOR = _IMG,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Img)
  ))
_sym_db.RegisterMessage(Img)

Img_rgb = _reflection.GeneratedProtocolMessageType('Img_rgb', (_message.Message,), dict(
  DESCRIPTOR = _IMG_RGB,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Img_rgb)
  ))
_sym_db.RegisterMessage(Img_rgb)

Circle = _reflection.GeneratedProtocolMessageType('Circle', (_message.Message,), dict(
  DESCRIPTOR = _CIRCLE,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Circle)
  ))
_sym_db.RegisterMessage(Circle)

Circles = _reflection.GeneratedProtocolMessageType('Circles', (_message.Message,), dict(
  DESCRIPTOR = _CIRCLES,
  __module__ = 'protos.dish.dish_pb2'
  # @@protoc_insertion_point(class_scope:dish.Circles)
  ))
_sym_db.RegisterMessage(Circles)



_DISH = _descriptor.ServiceDescriptor(
  name='Dish',
  full_name='dish.Dish',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=249,
  serialized_end=343,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetCircles',
    full_name='dish.Dish.GetCircles',
    index=0,
    containing_service=None,
    input_type=_IMG_RGB,
    output_type=_CIRCLES,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetImg',
    full_name='dish.Dish.GetImg',
    index=1,
    containing_service=None,
    input_type=_REQUEST,
    output_type=_IMG_RGB,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DISH)

DESCRIPTOR.services_by_name['Dish'] = _DISH

# @@protoc_insertion_point(module_scope)
