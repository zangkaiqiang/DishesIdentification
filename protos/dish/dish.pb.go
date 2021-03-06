// Code generated by protoc-gen-go. DO NOT EDIT.
// source: protos/dish/dish.proto

package dish

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	grpc "google.golang.org/grpc"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type Request struct {
	Name                 string   `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Request) Reset()         { *m = Request{} }
func (m *Request) String() string { return proto.CompactTextString(m) }
func (*Request) ProtoMessage()    {}
func (*Request) Descriptor() ([]byte, []int) {
	return fileDescriptor_a30b23922128f20b, []int{0}
}

func (m *Request) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Request.Unmarshal(m, b)
}
func (m *Request) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Request.Marshal(b, m, deterministic)
}
func (m *Request) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Request.Merge(m, src)
}
func (m *Request) XXX_Size() int {
	return xxx_messageInfo_Request.Size(m)
}
func (m *Request) XXX_DiscardUnknown() {
	xxx_messageInfo_Request.DiscardUnknown(m)
}

var xxx_messageInfo_Request proto.InternalMessageInfo

func (m *Request) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

type Response struct {
	Name                 string   `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Response) Reset()         { *m = Response{} }
func (m *Response) String() string { return proto.CompactTextString(m) }
func (*Response) ProtoMessage()    {}
func (*Response) Descriptor() ([]byte, []int) {
	return fileDescriptor_a30b23922128f20b, []int{1}
}

func (m *Response) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Response.Unmarshal(m, b)
}
func (m *Response) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Response.Marshal(b, m, deterministic)
}
func (m *Response) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Response.Merge(m, src)
}
func (m *Response) XXX_Size() int {
	return xxx_messageInfo_Response.Size(m)
}
func (m *Response) XXX_DiscardUnknown() {
	xxx_messageInfo_Response.DiscardUnknown(m)
}

var xxx_messageInfo_Response proto.InternalMessageInfo

func (m *Response) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

type Img struct {
	Data                 []int32  `protobuf:"varint,1,rep,packed,name=data,proto3" json:"data,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Img) Reset()         { *m = Img{} }
func (m *Img) String() string { return proto.CompactTextString(m) }
func (*Img) ProtoMessage()    {}
func (*Img) Descriptor() ([]byte, []int) {
	return fileDescriptor_a30b23922128f20b, []int{2}
}

func (m *Img) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Img.Unmarshal(m, b)
}
func (m *Img) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Img.Marshal(b, m, deterministic)
}
func (m *Img) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Img.Merge(m, src)
}
func (m *Img) XXX_Size() int {
	return xxx_messageInfo_Img.Size(m)
}
func (m *Img) XXX_DiscardUnknown() {
	xxx_messageInfo_Img.DiscardUnknown(m)
}

var xxx_messageInfo_Img proto.InternalMessageInfo

func (m *Img) GetData() []int32 {
	if m != nil {
		return m.Data
	}
	return nil
}

type ImgRgb struct {
	Img                  []*Img   `protobuf:"bytes,1,rep,name=img,proto3" json:"img,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ImgRgb) Reset()         { *m = ImgRgb{} }
func (m *ImgRgb) String() string { return proto.CompactTextString(m) }
func (*ImgRgb) ProtoMessage()    {}
func (*ImgRgb) Descriptor() ([]byte, []int) {
	return fileDescriptor_a30b23922128f20b, []int{3}
}

func (m *ImgRgb) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ImgRgb.Unmarshal(m, b)
}
func (m *ImgRgb) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ImgRgb.Marshal(b, m, deterministic)
}
func (m *ImgRgb) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ImgRgb.Merge(m, src)
}
func (m *ImgRgb) XXX_Size() int {
	return xxx_messageInfo_ImgRgb.Size(m)
}
func (m *ImgRgb) XXX_DiscardUnknown() {
	xxx_messageInfo_ImgRgb.DiscardUnknown(m)
}

var xxx_messageInfo_ImgRgb proto.InternalMessageInfo

func (m *ImgRgb) GetImg() []*Img {
	if m != nil {
		return m.Img
	}
	return nil
}

type Circles struct {
	X                    float32  `protobuf:"fixed32,1,opt,name=x,proto3" json:"x,omitempty"`
	Y                    float32  `protobuf:"fixed32,2,opt,name=y,proto3" json:"y,omitempty"`
	R                    float32  `protobuf:"fixed32,3,opt,name=r,proto3" json:"r,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Circles) Reset()         { *m = Circles{} }
func (m *Circles) String() string { return proto.CompactTextString(m) }
func (*Circles) ProtoMessage()    {}
func (*Circles) Descriptor() ([]byte, []int) {
	return fileDescriptor_a30b23922128f20b, []int{4}
}

func (m *Circles) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Circles.Unmarshal(m, b)
}
func (m *Circles) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Circles.Marshal(b, m, deterministic)
}
func (m *Circles) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Circles.Merge(m, src)
}
func (m *Circles) XXX_Size() int {
	return xxx_messageInfo_Circles.Size(m)
}
func (m *Circles) XXX_DiscardUnknown() {
	xxx_messageInfo_Circles.DiscardUnknown(m)
}

var xxx_messageInfo_Circles proto.InternalMessageInfo

func (m *Circles) GetX() float32 {
	if m != nil {
		return m.X
	}
	return 0
}

func (m *Circles) GetY() float32 {
	if m != nil {
		return m.Y
	}
	return 0
}

func (m *Circles) GetR() float32 {
	if m != nil {
		return m.R
	}
	return 0
}

func init() {
	proto.RegisterType((*Request)(nil), "dish.Request")
	proto.RegisterType((*Response)(nil), "dish.Response")
	proto.RegisterType((*Img)(nil), "dish.Img")
	proto.RegisterType((*ImgRgb)(nil), "dish.Img_rgb")
	proto.RegisterType((*Circles)(nil), "dish.Circles")
}

func init() { proto.RegisterFile("protos/dish/dish.proto", fileDescriptor_a30b23922128f20b) }

var fileDescriptor_a30b23922128f20b = []byte{
	// 210 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x8f, 0x51, 0x4b, 0x86, 0x30,
	0x14, 0x86, 0xdb, 0x37, 0xcb, 0x3c, 0xd5, 0xcd, 0x2e, 0x62, 0x15, 0x85, 0xec, 0x22, 0xbc, 0x08,
	0x03, 0xed, 0x1f, 0x14, 0x84, 0xb7, 0xfb, 0x03, 0xa1, 0x39, 0xe6, 0xa0, 0xa9, 0x6d, 0x0b, 0xf4,
	0xdf, 0xc7, 0x8e, 0xd6, 0xd5, 0x77, 0x33, 0xde, 0xe7, 0x3c, 0xec, 0xdd, 0x0e, 0x5c, 0xcf, 0x6e,
	0x0a, 0x93, 0x7f, 0xee, 0x8d, 0x1f, 0xf0, 0x28, 0x71, 0xc0, 0x92, 0x98, 0xc5, 0x3d, 0xa4, 0x52,
	0x7d, 0xff, 0x28, 0x1f, 0x18, 0x83, 0x64, 0x6c, 0xad, 0xe2, 0x24, 0x27, 0x45, 0x26, 0x31, 0x8b,
	0x07, 0x38, 0x97, 0xca, 0xcf, 0xd3, 0xe8, 0xd5, 0x51, 0x7f, 0x03, 0xb4, 0xb1, 0x3a, 0xaa, 0xbe,
	0x0d, 0x2d, 0x27, 0x39, 0x2d, 0x4e, 0x25, 0x66, 0xf1, 0x08, 0x69, 0x63, 0xf5, 0x87, 0xd3, 0x1d,
	0xbb, 0x03, 0x6a, 0xac, 0x46, 0x7b, 0x51, 0x65, 0x25, 0x7e, 0xa2, 0xb1, 0x5a, 0xc6, 0xa9, 0xa8,
	0x21, 0x7d, 0x35, 0xee, 0xf3, 0x4b, 0x79, 0x76, 0x09, 0x64, 0xc1, 0xfa, 0x83, 0x24, 0x4b, 0xa4,
	0x95, 0x1f, 0x36, 0x5a, 0x23, 0x39, 0x4e, 0x37, 0x72, 0xd5, 0x0b, 0x24, 0x6f, 0xc6, 0x0f, 0xec,
	0x09, 0xe0, 0x5d, 0x85, 0xbf, 0xfb, 0x57, 0xff, 0xd5, 0xf1, 0xd9, 0xdb, 0x1d, 0x77, 0x2b, 0x4e,
	0xba, 0x33, 0xdc, 0xbc, 0xfe, 0x0d, 0x00, 0x00, 0xff, 0xff, 0xb5, 0x09, 0x72, 0xc2, 0x13, 0x01,
	0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// DishClient is the client API for Dish service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type DishClient interface {
	GetCircles(ctx context.Context, in *ImgRgb, opts ...grpc.CallOption) (*Circles, error)
}

type dishClient struct {
	cc *grpc.ClientConn
}

func NewDishClient(cc *grpc.ClientConn) DishClient {
	return &dishClient{cc}
}

func (c *dishClient) GetCircles(ctx context.Context, in *ImgRgb, opts ...grpc.CallOption) (*Circles, error) {
	out := new(Circles)
	err := c.cc.Invoke(ctx, "/dish.Dish/GetCircles", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DishServer is the server API for Dish service.
type DishServer interface {
	GetCircles(context.Context, *ImgRgb) (*Circles, error)
}

func RegisterDishServer(s *grpc.Server, srv DishServer) {
	s.RegisterService(&_Dish_serviceDesc, srv)
}

func _Dish_GetCircles_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ImgRgb)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DishServer).GetCircles(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dish.Dish/GetCircles",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DishServer).GetCircles(ctx, req.(*ImgRgb))
	}
	return interceptor(ctx, in, info, handler)
}

var _Dish_serviceDesc = grpc.ServiceDesc{
	ServiceName: "dish.Dish",
	HandlerType: (*DishServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetCircles",
			Handler:    _Dish_GetCircles_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "protos/dish/dish.proto",
}
