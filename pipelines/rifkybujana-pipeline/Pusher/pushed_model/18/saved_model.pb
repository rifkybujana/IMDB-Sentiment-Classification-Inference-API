??
?-?-
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?N*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?N*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_377*
value_dtype0	
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name17613*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?)*
dtype0*??
value??B???)BtheBaBandBofBtoBinBisBiBthisBthatBitBbrBwasBforBmovieBwithBasBbutBfilmByouBonBhisBnotBareBheBitsBfromBatBoneBhaveBbyBbeBsoBorBaboutBallBjustBanBoutBifBveryBlikeBwhoBtheyBwhatBsomeBnoBhasBthereBevenBwouldBmeBgoodBtheirBmyBmoreBherBupBhadBseeBmostBwhichBwhenBwellBstoryBgetBdoBwatchBweBonlyBwhereBwayBthanBwillBwereBmadeBsceneBotherBbadB
charactersBgreatBbeenBtimeBreallyByourBthemBplotBpeopleBcanBwatchingBintoBtheseBfilmsBfirstBeachBdidBcouldBbecauseBtooBalsoBmuchBmoviesBendB	characterBsheBmakeBhowBnothingBbestBthinkBshowBlifeBactingBthoseBfunnyBdoesBbeingBanyBthenB	somethingBnewBbetterBthroughBscenesBimBdontBwhyBmanBbeforeBoffBneverBhimBfindBeverBrealBknowBgoBafterBwhileBsuchBstillBsayBsameBloveBhereBworldBlookBhoweverBcantBbigBtwoBthingBmanyBagainBworkBmusicBgameBdoesntBanotherBactorsBactionBwantBusBseenBownBnowBlotBfewBfactByearsBseemsBoldBmakesBlittleBdownB	differentBtheresBpointBbothBalmostBwithoutBthoughBquiteBplayedBonceBgoingBcastBaroundBamByoungBwholeBthingsBqualityBprobablyBoverBmayBkindBenoughBenjoyedBbetweenBbelieveBalthoughBthreeBthatsBsetBpartBmightBleastBkidsBgotBworthBwatchedBuseBthoughtBprettyBperformanceBlongBinterestingBgetsBfeelBcomedyBbookB	beautifulBworstBtakeBrestBreadBratherBpoliceBmustBknowsBiwerksBguyBfarBeveryBdirectorBdanceBcaseBbackBanyoneB	animationBamandaBalwaysBactuallyBwantsBsimplyBscriptBplaceBpictureBourBothersBmoneyBmenBmakingBkeepBhavingBgiveB
everythingB
especiallyBendingBeitherBdoneBcomesByetB	wonderfulBwhiteBtrueBterribleBsinceBshouldBshotBsawBpointsBpieceBmessageBleftBguysBgirlBff7BelseBdidntBcityBcharlieBawayBaudienceBwerewolfBwasntBwarBuntilBtrulyBstoriesBstarBsomeoneBskeletonBseriesBsenseBscreenBrightB	relationsBpoorBniceBnextBmrBminutesBmaybeBliveBitselfBideaBhimselfBfunBfoundBfanBfaceBduringB	certainlyBblackBawfulBattemptB
absolutelyByoureB
understandBtryingBtimesBstartBshortBsaidBrememberBplaysBperformancesBoriginalBnightBmichaelBlooksBleaveBisntBhesBhelpBheadBfinalBdvdB	directionBcomeBclassicBcameBboringBboBartB10ByesBwhaleBwasteBvideoBunderBtypeBtvBtownBtogetherBtakesBstreetBsquareBspecialBschoolBroleB
ridiculousBputB
particularBmyselfBlookingBletsBkillingBkevinBjuliaBjohnBhumanBhorrorBhomeB	hollywoodBgoesBgivenBfriendBfamilyBenixBeffectsBcgBbuildingBbringB	beginningBbeginBanythingBageBworseBwantedBviolenceBtitleBthemeBsurpriseBsureBstupidBsteveBstarsBsmallBsirBseveralBseemBseeingBromanticBratingBplayingBplayBperhapsBonesBoftenBnumberBnolanBmurderBmjBmainBleBlackBkilledBiveBinterestBinsteadBimdbB	hilariousBhalfBfurtherBfullB	extremelyB
experienceBexpectBexceptBentertainingBelementsBdisneyBdeadBdayBcoupleBcopBbecomeB
backgroundBamericanBagainstBactorB4BwrittenBwordsBwomanBwishBwifeBwesternBturnBtriedBtopBteamBtalentedBsuccessB
soundtrackBsortBsophisticatedBshowsBshesBseemedBscottBrelationshipB	recommendBproblemBpowerBohBmomentsBmindBmikeBmexicanBlevelBlearnBlastBknownBkillsB
introducedBintendedBhighlyBhighBgroupBfinallyBfemaleB	excellentB
eventuallyBentireBenglishBdullB	difficultB
differenceBdespiteBcrapBcouldntBcasbahBbudgetBbroughtBbehindBbecomesBbBalongBactualB3ByearByeahBwrongBwritingBwouldntBwordBwontBwomenBweirdBvisuallyBviscontiBvaluesBusedBtypicalBtriesBtittaB
throughoutBtheaterBtellBsupposedBsubtleBstuffBspeakBsoonBsongBsisterB	seriouslyBsequenceBrobertBrecommendedBreasonBpublicB
productionBprinceBpositiveBperfectBoverallBnoneBneitherBnearBmsBmotherBmeyersBmentionBmeaningBmatterBmatteisBmatteiBmajorityBlowBlivingBlinesBletBleavesBleadBlaughingBlaughBlaterBlateBkillersBkillerBjosephBjobBjackBislandBhitBhenryBgradeBgodBgivingBgeorgeBfocusBfinneyBfeltBfatherBfamiliarBexactlyBevilBeventsBenjoyBeastwoodBearlyBdresserBdramaBdirectedBdialogueBdiBdetailBdeathBdaysBcrimeB
completelyBcompleteBcheapBcharmBchanceBcameraBbitB
believableBattemptsBalreadyBallowBafghanistanBacceptB20ByourselfByorkBworksBwindowBwaterBwaitBviolentBviewerBusualBuniqueBunfortunatelyBturkeyBtortureBtimingB
thoroughlyBthinkingBtheyreBtexasBtellsBtalentBtakenBsuspenseBsubtlyBstudioBstrangeBstorytellingBstevenBstartsBspectacularBspaceBsoulB	skeletonsB
situationsB	situationBsinatraBshortsBsexualB	sequencesBscoreBscifiBsaysBsavingBsatisfactionBryanBrunBreviewsBresultBremindedB	questionsBprisonBpresentBpowerfulBpossibleBpersonalBpainfulBorderBnoteB	nominatedBneedBmovesBmostlyBmoralBmomentBmokoBmissBminorB	microchipB	memorableBmarriedBluckBlovedBlostBlookedBlondonBlivesBlineBlikableBlieBlaughedB	knowledgeBingridBillBhurtBhospitalBholdBheroBheartBhearBhappensBguessBgreatestBgracieBgivesBgirlsBgettingBgenreBfolksBflickB
flashbacksBfireBfilmbrBfavoriteBfashionBfantasyBfallBexpectedB	encounterBeasyBdutcherBdieB
definitelyBdecidesBdecentBdarkB	criticizeB	courtenayBcourseBconveyBcommentsBcinemaBchildrenBchildBcheesyBcheckBcauseBcaughtBcareerBburtBbunuelBblueBbarelyBavoidB	attentionB
apparentlyBanywayBannoyingBandrewsBalBadmitBactedBacrossBaboveB7B5B310B30B2BzeroByouveBwwiiBwriterBworkedBwittyBwithinBwalkBvillainBversionBveniceBvarietyBusuallyBusesBunlessBunfortunateBtwistsBturnsBtryBtroubleBtraditionalBtrackBtotallyBtotalBtookBtonyBthirdBthemselvesbrB
themselvesBthanksBterrificBtensionB
televisionBteenBtapeBtalkBsympatheticB
supportingBsuperfluousB
successfulB
subtletiesBsubjectBstunningBstrongBstayBstartedBspeedBsoulsB	somewhereB	sometimesBsoldiersBsoldierBslowBsitBsingleBsingingBshowingBshootBsheenBshareBshallBsexyBsetsBsentBselfBsecondBscaleBsaveBsadBrunsBrugglesBrevealBremindsBreleasedBrealityBreadingBratedBrainBpsBprotagonistBpromiseBprofessionalBprizeBpreviousBpredictableBpredatorB	portrayedB	politicalBpleaseBplayerBplacesBpiecesBpickBpcBpayBpartsBparticularlyB
originallyBolderB
occasionalBobviousBnovelBnormanBnorBnastyBmusicalBmundaneBmovingBmoviebrBmormonBmomBmissionBmineBmethodBmeetB	meanwhileBmeansBmaryBmainlyBlocalBlimitedBlikedBlightBlessBladyBkomodoBkingBkeptBkaneBjokesBjokeBjamesBitllBitbrBitalyBindeedB
impressionB
impossibleBieBicetBhugeBhouseBhopingBholesBheyBherselfBheroesBhellBheavyBhardBgoodnessBgeneralBfrolicsBfollowsBfollowBfineBfindsBfieldsBfeelsBfeelingBfastBfarcryBexperiencedBexampleBeveryoneBepisodeBentertainmentB	enjoyableBenhancesBendsBemployerB	emotionalBeasilyBdriveBdreadfulBdoubtBdoingBdisappointingBdetailsBdecidedBdealBcuteBcutBcubeBcriminalBcrewBcostsBcoolBconnectB
conclusionBconcernBcluesBcleverBclassBchoreographedBchinaBchannelBchangeBcgiBcertainBcarolBcannotBbuyBbrothersBbrotherinlawB	brilliantBboyBboredBbodyBbmovieBblowBbizarreBbetterbrBbeatBbeachBbasedBbankBbandB	audiencesB
atmosphereBasideBanimationbrBamongBamazingBaliveBadventBaddedB6BzipByoullByoudByoubrBwritersBwonderBwonB	witnessedBwineBwilsonsBwilsonBwilliamBwhoseBwhomBwhetherBwhatsB	watchableBwastedBvolumesBvividBviewersB	variationBvanBusingBusheredBurgeBunlikelyBunimaginableBunfoldsBunexpectedlyBuBtuneBtruthBtroublesBtripB	transfersBtoughBtoraBtimebrBtightlyBthinnessBtherebrBthankBtetsuyaBtermsBtellingB
technologyBtapesBtakingBsuspectB
supposedlyBsumB	sufferingBsucksB	stylishlyBstyleB
strugglingBstoodBsteroidsBstatedBstagesBstageBspokenBspokeBspendBsouthBsoundBsongsBsonBsomewhatBsolutionBsleepBsittingBsincereB
simplifiedBsimpleBsimilarBsilentBsignificantBsideBshownBshockBshipBshakespeareBsexBservedBseriousB	screeningBscottishBschnitzlersBscaryBsatansBsarahBsaltBsadlyBrunningBrubbishBroseBrosarioBrootsBrobinsonBrobBripoffBreviewerBreviewBrevealedBrepeatB
rememberedBremakeBreliesB	reexamineBredBrecordBrecentBreasonsB	reactionsBrarelyBrandolphBraceBquicklyBpurpleBprojectBproducedBproblemsBpreparedB	premingerBpregnantBpossiblyBportraitBplaystationBplanningB	placementBpetterBpersonBpatheticBparodyBparkerBpaintBpageBpacinoBpacingBoverdoneBoutsideBorientedB	organizedB	orchestraBopeningBontoBokayBoffersBoccasionallyB	obviouslyBnotchBnossiterBnormallyBnonethelessBnomuraBnicholsBneighborhoodBnearlyB	narrationBnameBmurderedBmoveBmotionBmorrisBmoonBmoodBmistakeBmissionariesBmethodsBmetB	mentionedBmentallyBmeanBmcqueenBmanagesBmajorBmaginnisB	luxuriousBlovesBlovelyBlotsB
lonelinessB	locationsBlistBliberalBleadsBleaderBlaneBlakeBlaBknewBkillBkeepsBjuliasBjourneyB
irritatingBinvolvedB
interestedBinspireBinhabitsBinconsistentBimprovementB	importantB	imperioliB
impeccableBiiBidiotBideasBhusbandBhotelBhorribleBhoodBhonestBhomeworkBhitsBhintBhersBheldBheardBhaventBhateBhappyBhappenBhalfwayBhadntB	habitatbrBgrenierBgreaterBgraphicB
governmentBgoreB	girolamosB
generationB	generallyBgaveBgangBfxBfulfillmentBfulcisB
friendshipBfrankBfourBforwardBforgetBfmvBfluidityBflicksBfitB
filmmakersB	filmmakerBfestivalBfellowBfearB	fantasticBfansBfairlyBfacialBfacesBfBeyesBeyeBextraordinaryBextraBexponentiallyBexperiencebrB	expectingBexamplesBexaminationB	everyonesBeveningBeraBepisodesBenterBendbrBencounterbrBemotionallyBemBeffortBearthBdyingBdumbBdrummerBdrugBdoubleBdoorBdollarsB
disturbingBdisneysB
disclaimerBdiscernsB	disbeliefBdisasterBdisappointmentBdirtyB	directorsBdiamondB	detectiveBdepthB
dependencyB	dellamoreB
deliverybrBdeliversBdeepBdearBdealtBdawsonBdancebrBdanaBcurseBcureBcryBcritiqueB	criminalsBcreatureBcreativeBcreateBcowBcoupledBcountryBcorruptBcontainsBcontactBconsequencesBconseguenzeB	connectedBcomicBcoldBcloseBclearlyBcivilBcinematographyBchurchBchicagoB	chemistryBcharmingBcharliesBcatchBcaringBcaptureBcalledBbusinessBbuscemiBbusBbunchBbringsB
brillianceBbrianBbrainBboysBbotherBbogardeBbluffBbitsBbeyondBbeginsBbeautyBbarBbadlyBawesomeBawebrBawaitB	availableB	attractedBattackB
associatesBarthurBarmBarlenBarkBarentBappropriateB
appearanceBappealB	apartmentB	anxiouslyB
animationsB	animatingBanimalsBanimalBaloneBallowingBalivebrBaliensBalasBagoBafghanBadrianBaddsB	accordingB	accompanyBaccidentallyB13B1BzzzzzzzzzzzzzzzzzzBzaitungByoungsByidischeByearsbrBwynorskiBwriteBwoundsBworkingBwoodsBwoodB	wonderingBwitchB	wilkinsonBwifesBwhosBwhitesBwhipBwheneverBwhalesBwestBwentBweightBwearsBweaponsBwealthyBweakBwashedBwarningBwarmBwalthallBwalkedBvoteB
voiceoversBvoiceBvividlyB	virtuallyBvillaBviewingBviewBvictimBviciousBvhsBvaletButterlyButterBurbanBunsympatheticB
unoriginalBunnecessaryB
unlikeableBunlikeB	unlikableBunintentionallyBunforgettableB
unexpectedBunderstandingBuncomfortableBunclesBunbornB	ultraevilB
ultimatelyBuglyBubBtyrantBtypesBtwistBtvmovieBturningBturdsBtrustedBtroubledBtrickB	translateBtrainBtrailerBtrailBtoylandBtownsBtowardBtouchedBtouchBtomBtoldB
togetherbrBtodayB	titillateBthrillerBthousandBthoBthisbrBthinksBtheydB	thereforeBthereaboutsB
theatricalB
thankfullyBtessaB
terroristsBtenB	techniqueBtechnicolorBtechnicallyBteachesBteacherBteachBtasteBtargetBtaleBsynthesizerB	symbolismBsweetBsuspectsBsurviveB
surrealismB	surprisedBsurgeryBsunBsummaryBsuckedBsucceedBstumblesBstruckBstrayBstraussB	storylineBstorybrBstoppedBstopBstoogesBstoneheartedBstoneBstixBstewartB
stereotypeBstepBstefanB	statementBstaringBstabbedBstBsportsBsportBspoilB
spectatorsB	spectatorBsouthernBsorryBsomehowBsocietyBsocalledBsoapsBsoapBsnowBsmithBsmartBsmarmyB	slowpacedBslightlyBsimonBsillyBsilkkBsidedBsickBshowedBshowbrBshotsBshorterBshopBshoeBshockedBsheerBsharpBsettingB	sensitiveBsensibilitiesBsendBselectBsegmentsBsegalB	seeminglyBsecretsBseasonBsearchBseanBsealBseaB
screenplayBscareBscandinavianB
satisfyingBsatireBsakeB	sacrificeBrussianBrourkesBroundBroughBrootingBroomBromanticismBromanceBrolesBrobertsBroadBriseBrichBreynoldsBrevolveB	revealingB	returningBrespectBrepresentedBrepeatedBrentingBrentedBreleaseB
relativelyBregularBregardedB	referenceBrebelBrealizedBrealismBreadyBreadsBraspyBracerBquotesBquietBquickBquibbleBquestioningBqinBputtingBpushBpureB	psychoticBpsychoBprovidesBprovideBprovesBproperB
progressesBprofoundBproductionsBprivateBprintBprimeBpriceB
previouslyB
pretendingBpresumeB
premingersBpovertyBpossibilityBpopularB
politicianBplentyBpleasureBplasticBpityBpileBpigsBpicturesBphotographerBpgB
personallyBperiodBperformBperfectnessBpepeBpeaceBpassBparkBparentsBparBpacedBpaceB	overratedBoutjustBottoBoscarBosamaBoriginalityBorcaBoppositeBopinionBoperasBopenBonewordBoliviaBokBofficerBofficeBoddsBoddB	occasionsBnumerousBnudityBnowadaysBnoticeBnonsenseBnonlifeBnominationsBnewsBneverthelessBnephewBnelsonBneedsBneedlessBnavyBnatureBnativeB	narrativeBnamedBnakedBmutatedBmusicianBmovementBmovedBmouthBmorningBmoranBmollyBmodernBmistakesBmissingB
miraculousBminsB
miniseriesB
minimalismBmiltonBmilitaryBmilesBmiddleBmessBmentalBmenaceBmemoryBmembersBmemberBmelodyB
meaningfulBmcgavinBmateBmasterpieceB
mastermindBmasterfullyB	marvelousBmartyBmartianBmarsBmarkBmannerBmanagerBmanagedBmaleBmakeupBmahlersBmahlerBmagicBmadgeBloversBloudBlossBlooseBlonnrotBlongerBlocationBloadBlloydBlimitationsB	lifestyleBliesBlessonBlegacyBledBleavingBlearningBlearnedBlearBleagueBleadingBleadenBldsBlaughablebrBlatterBlatinoBlargelyBlanguageBlameBlacksBkungBkongBkindsBkindaBkillingsBkidBkickingBkansasBjrBjonesBjonathanBjimBjeanBjazzBjasonBjapaneseBjapanBjamaicanBjadedBitalianBissuesBironyBinvestigationBintrigueBinternetBinternationalB	intentionBinsomniaBinsidesBingredB	influenceB
infidelityB
industrialBindianB
incredibleB	includingBincludedBincludeBimprovedB	impressedB
importanceBillegalBignorantBidentityBidBiceBhypeBhybridBhuntBhumorBhumansBhoursBhotBhostBhorridBhopesBhopelessB	hopefullyBhopeBhookedBhongBhonestlyBholdsBhistoryBhiredBhillsBhidingBhideousBherosBhelpsBhelpedB
helicopterBheistBheckBhartmanBharshB	happinessBhappilyBhappierBhappenedB	halloweenBhairB
guggenheimBguestsBguessingBguardB
guaranteedBgruesomeBgrowthBgrowsBgrowBgriefBgrewBgreenBgrandBgraduateB	graduallyBgradingBgordonlevittBgoodbrBgodzillaBgodsBgloverBglennBglanceBgladBgirolamoBgenuineBgemBgainBfundingBfulciBfuBfriendsBfridayBfrickerBfreshBfrequentBfrenchBfreedomBfreeBfreakyBfranzBformB	forgottenBforgettableBforeignBfordBfongB	followingBfoilBfloorBflippinBfleshBflatBfinishB
filmmakingBfilmingBfilmedBfellowesBfeetBfeelingsB	favouriteBfascinatingBfarceBfamousBfallingBfakeyBfakeBfailureBfailedBextrasBextractBextentB
explosionsBexplicitBexplainsB
explainingBexpectationsB	existenceB	excusableBexclusivelyBeventBeuropeanBeuropeBetcbrBetcBetBessentiallyBerkanBenteringB	enjoymentBenglishsBenergyB	encourageB	empathizeBemotionlessB	elizabethBelementBelegantBeleanorBeightBeffectBeerieBeditingBedBearlierBdynamicBdueBdudeBducklingBdryBdrivesBdreamsBdozenBdoyleBdoomedBdoomBdomainBdogsBdogBdocumentaryBdocumentBdistanceB
disfiguredBdiscreetB	discoveryB	discoversBdiscoverBdisappointedB	directingBdialogBdevicesB
developingB
determinedBdeterminationB
destroyingBdestroyBdesireBdesertB	describedBdescribeBdescendsB	depictingBdemiseB
deliveringB	delightedBdegreeBdefiniteBdeeplyBdeeperBdealerBdeBdawnBdavisBdavidBdarrenBdareBdanzaBdannyBdanglingB	dangerousBdancingBdamnedBdammeBdadBcutsBcuriousBcultureBculturalBcruelBcrossBcrooksBcrookedBcrippledBcreditB	creaturesBcreatorsBcreatedBcrazyBcrashBcrankBcrackingBcoxBcoversBcoveringB	countriesB	countlessBcountingBcountessBcountBcostelloBcostarsBcostBcornerBcopyBcopsB
coproducedBcooperBcontributorsB	continuedBcontextBcontentsBcontentBconstructedBconsiderationB
connectionBconfusedB
confidenceB
conditionsBconcernsB	concernedB	conceivedBconBcomposedBcomplexB
complaintsBcompilationBcompetitionB
compassionB	communityB
commentaryBcomicalBcomediesBcomedicBcomedianBcombinationBcolorfulBcolbertBcohostBcoBclumsyBclothesBclipsBclintBcliffhangerBcliffBclearBclayBclassicsBclassicallyBclarkeB
chupacabraBchunkBchristopherBchoseBchoreographyBchooseBchineseBchickBcheatsBcharlesBcharismaticBchaplinBchaosBchangesBchangedB	challengeBcattleBcatsB	catholicsB	catherineBcatBcasesBcartoonBcarsBcaresBcareersBcarBcapitalBcanadaBcalleiaBcallanBcallB
californiaBbushBburnsBburningBbullyB	bulgakovsBbudgetsBbrotherBbroadBbridgeBbrendanBbrendaBbreakingBbrandBboylesBboyerBboxBboundBboughtBbottomBborgesBbolandBbogardesBboatBboardBbloodyBbloodBblockbusterBblinkBblindBblameBblacksBbirthBbillBbiggestBbesidesBbentonBbennyB	believingBbeganBbedBbecameBbeautifullyBbeatingBbeatenBbattlingBbatmanBbasicBbaseballBbaronBbarbaraBballBbadbrBbabyBbabesBawhileBawardsBaverageB
attractiveBattachedBassumeB
assistanceBasksBaskedB
appreciateBappearsB	appearingBapartBannaBanimeBanimatedBancientBanatomyB
amateurishBamateurBalmightyBallbrBalienpredatorBalienBaimedBaidedBaidBafraidB	aestheticBadviceB	advantageBadultsBaddictedBadaptedBactionsBactB
accustomedBaccountBacceptedBaccentBacademyBacB	absurdityBabsenceBabilityBabbottB9yearoldB810B70sB50B2005B1984ishB1981B1978B1959B1929B15B12B1010BzombiesBzombieinfestedBzombieBzapperBzanyBzaneByoutubeByouimByorkerB	yesterdayByatesBxxxxxxBwyomingB	wynorskisBwwwaaaaayyyyyBwroteBwritesBwriterdirectorBwreckbrBwreckBwowBwovenBwoundB
worthwhileB	worthlessBworthbrBworseandBworryBworriedBworldsBworldbrBworksbrBworkmanlikeBworkbrBwoodlandBwobblyBwitnessB
withdrawalB
witchcraftBwishesBwiseBwinnersBwingBwinBwillingnessBwillingB
williamsonBwilliamsBwillesBwileBwildlyBwildeyedBwildBwidewasBwiderBwhorishBwhoopiBwhipbrBwhinneysBwhinneyBwhineB	whimsicalBwhilstBwhereofBwhereasBwheedleBwhateverBwhalingBwhalehuggerBwhalecharacterBwerentBwenB	wellactedBwelcomeBweddingBwearingBwearBweaponryB
weaknessesBwcBwaysBwaversBwaveBwattdecidesBwatsonB	waterstonBwatchforBwatchersBwatchedmoviesBwatchbrBwastesBwashBwarsBwarnabysBwarmingBwardsBwardenBwardBwantingBwannaBwandersBwandaBwaltBwalshBwallsBwalledBwallBwalksBwalkingBwalkerB
waitressesBwaitressBwackyBwBvulgaritybrBvulgaritiesBvsBvomitB
volunteersB	volcanoesBvolcanoBvoicesespeciallyBvoiceofreasonBvixenBvisualBvisitorB	viscontisB	viridianaBvioilentBvincentBvignetteB	viewpointBviewedBvideosB	victorianBvicariouslyBveteranBvesselBvertigoBversusBversionsB
vernacularBvergeBventB
vendettabrBvendettaBvehicleBvegasBvcrB
vaudevilleBvastlyBvalueBvaluableBvalidBvalerieBvalbrButteringBusersBusbrBusageBupperBuponBupdatesBunwatchableBunsuspectingBunsupervisedBunstrungBunsafeB
unrestoredBunremarkableBunrealisticallyBunrealisticBunqualifiedB
unpreparedB
unpleasantBunoriginalityBunorganizedBunmitigatedBunmarkedBunluckyBunknownB
universebrBuniversallyBunitedBunionBuninterestingBuninspiringB
uninspiredBuniformsBunharmedBunhappyBunglamorousB	unfoldingBunevenBundrBundoneBundesirableBunderstatedBunderstandablyB
underneathBunderhandedBundercuttingBunderappreciatedB
undeniablyBuncutBuncurseBunconsciousBunclearBunbelievablyBunbelievableBunappealingBunanticipatedBukBtyroneBtwisterBtwistedBtwentysomethingBtwentyBtvshowsB	tururroisBturnedBturgidBtunesBtrustBtruckBtroopsBtrivializesB
triumphingBtrioBtributeB
tremendousBtrekB	treatmentBtreatedBtreatBtravelsBtraumatizedB	traumaticBtraumaBtrapbrBtransportedB
translatorBtranslationB
transitionB
transformsBtransformationB
transcendsBtranscendingBtraitsBtrainedBtrailridersBtrailingBtragicBtraceBtownsmanBtowelBtowardsBtouristBtoupeeBtoungueincheekBtouchesBtossB	tornadoesBtornBtorchBtopsBtopofthelineBtopicBtop10B
tonalitiesBtokyoBtoiletBtoetapBtodaysBtittasBtitsBtitlewritingBtissuesBtiringB
tiresomelyBtipBtinyBtinaB
timeswhichBtimerB	timemoneyBtimelessBtimBtillBtiesBtierneyBtiedBtidalBtianxiachineseBthumbBthrustBthrowsBthrowBthroughoutbrB	throughlyBthrillertypeB	threatensBthreateningB	thousandsBthoughtprovokingB	thoughtbrBthomasBthisheBthinlydisguisedBthingsbrBthinB	thighwithBthievesBthiefBthickBthemesBthematicBtheftBthebrBtheatricallyBthatbrBtexturedBtextB	terroristBterrorisingBterrorBtermBtentativelyBtendBtemplesBtelegramBteenageBtediousBtechnologyoutB
techniquesBtechnicianbrBtechBtearsB
tearjerkerBteambrBtayeBtawdryBtaughtB
taskmasterB	tarnishesBtargetsBtannerB	tanktruckBtangentBtalksBtalkedBtalibanBtalentsBtadzioBtacklesBtaciturnBtabooBtableBsystemBsymphonyB
symphoniesB
sympathizeB	symboliseBsymbolicBsykesBswitzerlandBswitchedBswitchBswissBsweptBswellingBsweatyBsweaterBswearsBswearBsvuBsuyinBsuvB
sutherlandB
suspicionsB
suspensionBsuspensefulB	suspendedBsuspendB	survivorsB	survivingB	surroundsB
surrealistBsurrealBsurfaceB	suprisingBsupposeBsupportBsuperstardomBsupernaturalBsupermarketB	superiorsBsuperblyBsuperBsunnyBsundryBsumsB	summingupBsummerBsuitcaseBsuitableBsuicideBsuggestsB
suggestionB	suggestedBsuggestBsugarcoatingB
sufficientBsufferedBsufferB
suddennessBsuddenlyBsuddenBsucksbadBsucceededbrB
subversiveB	subtitlesBsubstandardB	substanceBsubplotsBsubplotB	submittedB	sublimelyBstuttersBsturmunddrangBstupiderBstuntB
stunninglyBstuffyBstuffbrBstuckB	struggledBstruggleBstronglyB	strongarmBstrokeBstringBstrictlyBstretchprintedB
stretchingBstretchBstreamBstraightawayBstraightBstoutBstorytheBstorysBstoreBstopsBstolenBstoleBstingingBstillslinkyBstevesBstevennathanBsternerBsternBsterileBstereotypicalBstereotypesBstereotypedBstereoBsteppedBsteerBsteamingBstealsBstayedBstationsBstathamBstatesBstateoftheartBstateBstartingBstartersBstarringBstarredBstarklyBstarkBstaresB	stanwycksBstanwyckBstandupBstandsBstandingB	standardsBstandBstalwartB	stallingsBstallingBstageyBstaffBstackedBstabsBsquealB	squabblesB	sprawlingB	sportsmanBspokebrBspoilersBspoilerBspoiledBsplitB
splendoredBspleenbrBsplashBspiritB	spielbergBspendsBspeltBspellB
speechlessBspeechesBspecsBspecialreallyB	specialbrBspeakingBspatB	sparklingBsovietB	southwestBsouthernersBsoundsBsoughtBsosoBsorrybrBsorrowBsorkinBsoreBsoonerBsolidBsolesBsoldBsoftcoreBsofiaB
sodbustersBsodB	socialiteBsocialBsobrBsnuckB	snorefestBsnoopsB	snatchersBsnappyBsnapBsmoothBsmokingBsmilesBsmellB
smartaleckB	smalltimeBsmackedB
slowpacingBsloppyBslimaneBsliceBslewfootBsleepsB
sleepinessBsleazeBslashesBslasherhorrorBslappingBskipsBskipBskinbutBskillsBskillfulBskiesBskepticsBsizesBsixtiesBsixBsitebrBsiteBsitcomBsinisterBsingletsBsingerBsinatrasBsimultaneouslyB	simperingBsimoneB	similarlyBsimilaritiesB	silvestroBsilverBsilliestBsignificanceBsignedBsignBsiegelsBsiegelBsidjohnB	sidewalksBsidewalkB	sickeningBsicBshtBshrugB	shriveledB
shrillnessB	shreddingBshowiBshouldntBshotgunBshortenBshortcomingsBshootsB	shootoutsBshootoutBshockersBshiteBshirleyB
shipwrecksBshiningB	shihuangsBshihuangmeansBshiftsBsheriffBshenBsheepleBshearsBsharplyBshapesBshamelesslyB	shamelessBshameBshakyBshakespearebrBshakespeareanBshakenbrBshaggyBshaggingBshadowsBsfcomedyBseymourBsexuallacedBseveredBsettledBsettingsBsethisBservicesBserveB
serpentineB	sequelsbrBsequelB	septemberBseptB	separatesBseparateBsentimentalBsensibleBsensesBsensebrBseniorBselmaBsellerBsellBselfrighteousBselfishBselfinflictedBselfindulgenceBseethingBseesBseemingB
seductressBsectionsBsecretBsecondsBsecondlyBsecondbrB	secondaryB	seatedgedBseatB	searchingB
seamonsterBsealsBseagalBseadrowningBscumB	sculptureBscrupulouslyBscroatBscriptwritersBscriptsBscriptedBscreensBscreenplaybrBscratchBscowlingBscouringBscorseseB	scorewithBscoresBscifiactionBschoolsBscholarsBschmuckBschemingBscenarioBscarredBscarfaceBscarfBscaredBsayingBsaybrBsavvyBsaversBsavedBsaucyBsaturdayB	saturatedBsatireparodyBsatBsappyBsansBsanderBsandBsamBsalernoBsalazarBsaidbrBsadnessBsadisticB
sacrificedB
ruthlesslyBruthlessBrutBrusskiesBruralBruleBruinedBruinBrubeBrpgBrourkeBroundtreebrB
roundaboutB	rotationsBrosieBrosenbergbrBropeBrootedBrooftopBronaldBrompsBromanticallyBromancesBrollBrocketsBrobotsBrobinBrobberyBroasarioBrkoBrivotingBriverBrivalBritalinaddictsBriskBrippingBripleyBriotBrinoBringwaldBringsB	righthandBrightandBridingBridiculouslyBriderBriddenBrickBrichieB
richardsonBrichardsandBrichardB
revolutionBreversalBrevengeBrevealsB
revealedbrBrevBreusedBreturnBretinalBresumeBresultsBrestrictiveBrestoredBrestoreBresponseB	respondedBrespectsB	residentsB	residencyBresearchBrescueBrequiresBrequiredB
reputationB	repulsiveB
repressionBrepresentativeBreportsBreporterBreportedBreplyBrepleteBreplacementBreplacedB	repeatingBrentingbuyingwatchingBrentBrenewalBrenaissanceBremovingBremoteBremnantsBreminiscentBrememberingBremainsB	remainderBremainbrBremainBrelyB	reluctantB
religionbrBrelievedBreliefBrelaxingB	relativesBrelativeBrelationshipsBreincarnationBrehashB	regularlyBregretsBregretBregionsBregimeB	regardingBregardBrefuseBrefugeeB
refreshingBreformattedBreflexesBreflectsBreflectBrefinedB	referringBreferredB
referencesBreeksBreedB	redundantBredneckB	redeemingB	recruitedB	recreatedBrecreateBrecoverB
recountingBrecordbreakingB
recognizedB	recognizeBrecklessBrecipeBrecentlyBreceivedBrecBrebelsB
reasonstheBreasonbrBreamsB
reamainingBrealizesBrealizeB	realitiesBrealisticallyB	realisticBrealiseBreaderBreachingBreachesBreachedBraymondB	ravishingBraveBratingsBrarefiedBrapscallionBrapportBrapistBrapidlyBrapedBrangersBrangeBrandomlyBrandomBranchersBrancherBranBramsayBrammedBrakishBrailroadBraidedBracismBrBquirkyBquinnBquietlyBquestionB	qualitiesBpyramidsBpushingBpursuitBpurringB	purposelyBpurezB	psychosbrBpsychosBpsychobitchBpsychiatristBpseudointellectualsB	protégéB	prototypeB
protestersB
protectiveBprotectBprotagonistsBprospectorsBprosaicBpropsBprophetB
propellersB
propagandaB
pronouncedBpromptlyB	promotingBpromoBprojectsB
projectingBprogressesbrBprogramsB	profuselyBproductsB	producersBproducerB
proclaimedBproceedsBproceedingsB	procedureBproBprintingBprintedBprincesBpriestBpreviewBpreventBpretentiousBpretendsB
presumablyBpressB
presentingBpresenceB	preschoolBprerecordedBpreposterousBprepareBpremiseB	pregnancyBpreferBprefaceBpredominantlyBpredicamentsBpredatoralienB
prebeverlyB
practicingB	practicesBpracticeBpracticallyBpowellB	powderkegB	potentialBpostwarBposterBpostBpossibilitiesBportraysB
portrayingBportrayBportionBportingB
portentousBpornB
popularityBpopulaceBpoorlyBpoolBpollyBpoliticsBpoliteB	policemanBpokingBpointlessbrB	pointedlyBpoignantB	plowrightBplotsB	plotlinesBploteverythingBplightsBpleeeaseBpleasantB	plausibleBplantsBplanetbrBplanedBplanBplaguesBpivotalBpitifulBpithardBpitchBpiritesBpiratesBpinpointBpinkhamBpinkBpillowBpiecesbrBpicksB	physicianB
physicallyBphotographedBphoneyBphoneBphillipsB	phillippeBphilipBphatBpharohsBpfeifferBpetulantBpettyBpeterBperverseBperspectivesBpersonalityBpersonaBperkyB
performingB
performersBperfectoB	perfectlyBperfectionbrB
perfectionBpereiraBpereBperceptionsB
perceptionBpeoplesBpeoplebrBpensiveBpeddlingBpeddlerBpeculiarBpeasantsBpeakB	peaceableBpaymentBpayingBpaulBpattonBpatternBpatriarchalBpatientsBpathBpasséB
passionateBpassionBpassesB	passengerBpassedBpassableB
partygoersBpartnerBpartingBparticularbrBpartialBparryBparodycomedyBparisBparapsychologyBparamourB	parallelsBparadeBpanBpamelynBpalmaBpalestinianBpalBpaintsBpaintersB	paintedonBpainBpaidBpackBownerBoverviewB	oversightB
overrunnotBoverrunBoverlyBoverlookingB
overlookedB	overjoyedBoveritsBoverflowingB
overdonebrB
overactingBoutfitsBoutfitBoutcomesBoutcomeBoutaB	ourselvesBoughtBotakuB
ostensiblyBosheaBorwellBoriginsBorganiseB
orchestralBorangesBopticalB
oppressiveB
oppressionBoppositenessBopinionsdontBophulsBoperatesBoperaBopensBonstageBonscreenBonewellBonedimensionalBolympicBoliversBoliverBoldfashionedBolB
offputtingB	offeringsBofferedBofferB	offensiveBoffendedBoffendBoffbeatBodonnellBoddlyBoceanBoccursB	occurringB
occurrenceB
obsessionsB	obsessionBobservationB	obscurityBobscureB	objectiveBobjectBnycBnyBnumbersBnprBnowhereBnovaksBnovakB	notoriousBnotionBnotesBnotedBnotbrBnotashotBnotablyBnosethumbingBnorthBnormalcyBnonstopBnonsplatterB
nonliteralBnonexistentBnondocumentaryBnondescriptB	nonactorsBnonB
nominationBnoirBnodoubtBnobudgetBnobleBnixBnineBnightsB
nightmaresB	nightmareBnightlyBniftyBnieceBnickyBnicerBnewsregrettablyBnewellBneuroticBneurosisBnetherlandsBnetBnerdyB
neorealistBneoBnelsonbrBneldelBneighborBneededBnedBnecessarilyBneatlyBnearperfectBnearestBnationB	nathanielBnancyBnamesbrBnamesBnaiveBnabBmythBmysteryB
mysteriousB	mysteriesBmutilateBmuteBmusterBmustacheBmuslimBmurrayBmurphyBmurdersBmuppetbrBmunnBmunichBmumbledBmultipleBmultibillionB
mujahideenBmuggingBmuddyBmudBmuchbrBmtvBmst3kBmrsBmovieweBmovietheB
moviesteerBmovieohBmovieespeciallyB	moviedontBmouthedBmouseBmouldsBmotivesB
motivationBmotivatedbrBmotionsBmothersBmotherinlawBmostbrBmoronsbrBmorebrBmooresBmontilleBmontanaBmonsterB
monkeylikeBmonkeyBmommaBmomentumBmolinaBmodifiedB
moderatelyBmodelBmockumentaryBmobstersBmixedBmixB
mistakenlyBmistakenB	mistakebrBmissionsBmissesBmissedB	misplacedBmisogynisticBmishapsB	misguidedB	miserablyB	miserableBmiscastBmirrorBmirandaBmiracleBminogueBminneapolisB
minluckilyBmindbrBmincingBmildBmilanBmikeyBmiguelBmightyBmidstBmidperformancebrBmidnightB	middlemanB
middleagedBmid20sB
microchipsBmickeyBmichelleBmichelangeloBmiamiBmi21B	metaessayBmessyBmessagesBmessageboardB	messaboutBmesmerizingB
mesmerizedBmerrillBmeritsBmerelyBmentionsB
mentioningB	mentionbrBmensBmenbrBmenageatroisBmemorialB	melvinlouB	melvillesBmeltingBmeltB	melodramaBmeldsB	melbourneBmelangeB
megacheeseBmegBmeetingBmeercatB
mediocrityBmediocreBmedicalBmediaBmebrBmeasureBmeantBmeaninglessBmeaningfulnessBmeagreBmcqueensBmcnallysBmcclainBmccarthyBmcburneyBmcarthurBmaturedBmatrixesqueBmatrixBmaterialBmastrantonioB	mastersonBmasterpiecesBmasterBmassiveBmassesBmassBmartineBmartialBmarringbuyingBmarlonBmarketBmarkedB
marginallyB	margaritaBmarcelBmarathonBmaranBmanwhaleBmanseBmansBmannersB
mannerismsBmannaBmannBmankindBmaniacBmanageBmalesB
malcontentBmalaiseBmakhmalbafsB	makeshiftBmakerBmajidiBmajestyBmaintainingB
maintainedB
mainstreamBmainlandB	maimingbrBmailBmagnaniBmagicalBmaggotBmafiaB	machinistBmachinationsBmacgyverBmacdowellisB	macdowellBlyricsBlyricBlyleBlustBlurkingBlufkinBluckyBluckilyBluchinoBloyalBlowsaveBlowellBloweBlovingB
lovemakingBlovebrBlouiseBlouisbrBlouisBlotbrBlostbrBlosesBloseBlordbrBlordBloosingB	lookingbrBlookinBlongtimeBlongsufferingBlongoriaB	longhornsBlongestBlogicalBloggiaBlockedB
locationbrBlocatedBlobBloadedB	livestockB
literatureB	literallyBlitaBlistensBlistenBlistedBlisaBlinkingBlineupB	limousineBlimitBlimbalsamatoreBlikewiseBlikesBlightsBlightlyBlightingBlightandBlifelongBlifelessBlifeiBlifebrB	lidzandieBlidzB
licentiousB
librariansB	libertineBlibertarianBlffBlevelheadednessBlettersBletterBletdownBlesserbutknownBlesserB
lesbianismBleopoldBleoB
legitimateBlegislationBlegendsB	legendaryBlegendBlegBleeB	lecherousBlawrenceBlavaBlaurenceBlauraBlaughterBlaughsB
laughogramBlatinosBlatestBlatentBlateeshaBlatebrBlastingBlastedBlasBlarsBlargerBlargeBlandsBlandladyBlandisBlandBlamarrB	lafayetteBladenbrBlackingBlackedBkylieB
krzyzewskiBkriegerBkomanbraBknucklesBknockingBknockBkitchensBkissedBkingsBkingdomBkimBkillaBkiefersBkieferBkidnapBkicksB	kickboxerB
kiarostamiBketchupB	kerbridgeBkendisBkellyBkeithsBkeithBkeatonBkazanBkabulBjusticeBjuryBjunkieBjunglesBjumpingBjumpBjuliabrBjubalBjoyfulBjoyBjournalistsB
journalistBjosieBjosephsBjokeyBjokersBjoiningBjohnsonBjohnsBjoelmikeBjoeBjodyBjockBjoanneBjoanBjjlBjetblackBjetBjeromeBjerkyBjerkBjenniferBjawslikeBjasonsBjarringBjaredBjanvierBjailbrBjailBjaguarsBjacksonBjackieBjBivyBityoullBitiBitalysBitalianspeakingBissuedBislandsB	irritatesBirresponsibleB
irrelevantB
ironicallyBiranianBinvolvesBinvolveBinvestigatorBinvestigationsBinvestigatingBinvestigateBinventorBinvasionB
invariablyBintroductionBintroducingB	intricateB
intimatelyBintimateB
interwovenBintervieweesB	interviewBinterventionBintersectionBinterruptionsBinterruptingBinterruptedBinterpretationBintercutBinteractionBintentBintelligentB
instrumentBinstantBinstallmentBinstallationsBinspiredBinspirationB	inspectorBinsignificantB
insightfulBinsideBinnuendoB
innovativeBinnocentB	innocenceBinjuriesB	initiallyBinitialBingredbrBingameBinfrastructureBinferiorBinfamousBineffectualBindustryBindoctrinatedBindividualsB
individualBindifferenceBindieBindicateBindiaBindescribablyBindefinableB
incrediblyBincreasinglyB
incoherentBincludesB
inclinedbrB	incidentsBinchBincestB
incendiaryBincB	inanimateBinaccuratelyBimproveB
improbableB
impressiveBimpotentBimportantlynotBimplicationsBimplicatingB	implicateB	implantedB
imperviousB
immigrantsB	imitationBimhoBimaginationB	imaginaryB
imaginableBimagesBimageBillnessBillicitBignoringBignoreBidioticallyBidioticB
identitiesBidentifyB	identicalBidealBicilyBicesBhystericallyaBhystericallyBhypotheticallyBhypnotisingB	hypnotiseB	hypercubeBhusbandsBhurtsBhuntingBhunterBhunkyBhumourhopingBhumourbrBhumorousBhummingBhumiliatingBhullBhuhbrBhughesBhuggersBhuggedBhugelyBhubertbrBhoytB	howsoeverBhoustonBhoundsBhotdBhostsBhostileB
hospitalorBhorseBhorrorsuspenseBhorrorslasherBhorrorscifiBhorrorsB	horrifiedBhorizonBhopperBhoopBhookeditBhonorBhomogenisationBhomageBholdenBhoffmanBhittingBhithertoB
hitchhikerBhissesBhisherBhisbrBhintedBhindsightbrBhindiBhimevenBhimbrBhilltopBhikesBhightechB	highlynotBhighlightingBhighestBhiddenB
hicksvilleBheyheytwisterB	hestonyulBhesitantBherringBheroinBheresBherebrBherbigBhenceBhelplessBhelpingBhelmetBhelloBheedBhedyBhechtBheavyweightBheavenBheartsBheartilyB	heartfeltBheartedB	heartbeatBhealthBheadsBheadedBheadacheBhaydensBhauntBhatredBhatfulBhaterBhatefulBhatedBhastenB
hasslehoffBhasntBharwoodBharveyBharrisBharlanB	hardwickeBhardenedBharborB	happensbrB	happeningBhansBhanksBhangupsBhangingBhangB	handsdownBhandsBhandleBhandicappedBhandheldBhandfulBhandedBhandbrBhandBhanBhammingBhammerBhaltBhallucinationsBhalfthoughtBhalfhourBhairlessBhairdyeBhahahBhaggardB	hackneyedBhackettBhabitsBhabitBguybrBguyandBgutterBgutsB	gutierrezBgustavBgustBgunshipsBgunsBguiltyBguidesBgrudgeBgrownBgrowingBgroupsBgroupieBgroundbreakingBgrossBgrippingBgripeB	griffithsBgreysBgreyBgrenadesBgreedyBgreatlyB	graveyardBgravesbrB
gratuitousB
gratifyingBgratefulBgrandmothersBgranddaughterBgraciesBgraceBgrabbedBgpssBgothicBgotchaBgoshBgorebrBgoonsBgoodsBgoodbyeBgonnaBgoldenBgoldbergBgoddontBgodbrBgoalBglowingBglobeBglobalisationB	glaringlyB
glamorizesBgiraffeBgintyBgingerB	gilligansBgiglioBgiantBgiallomurdermysteryBgialloBghettowhichBgestureBgesticulatesBgermanB	geraldineBgeographicallyBgenrebrBgenitalsBgeneticallyBgenerousBgeneBgeekyBgeekBgayBgathersB	gatheringBgaryBgardenvarietyBgardenBgarbledBgarbageB	gangstersBgangsterBgamesBgamerBgamebrBgamblersBgagsBgagBgabyBgabrielaBgabrielBgBfutureBfussingBfurthermoreBfurloughBfunneledBfunkB	funfilledBfundBfumesBfullyBftwBfrustrationBfrustratingB
frustratesBfrontBfrogBfrighteningB	friendsbrBfriendbrB	freshnessBfredBfreakedBfrankenheimerB	franchiseBfrancesB	frameworkBframesBframeB
foundationBfosterB	fortunebrBfortuneBforteBformulasBformulaBformingBformerBformedBforgivenessBforestB
forerunnerBfordsBforcedBforceBfootballBfooledBfondlyBfondaBfollowedBfolkBflyingBflushingBfloridaBfloraBfloozyBfloodBflipsBflippingBflipB	fleischerBfleetingBflawlessBflawBflatironBflashBflaringBfkedBfkBfiverBfiveBfitbrB
fistfightsBfishyB	fishermanB
fingertipsBfingersBfingerprintsBfindingBfinBfilsBfilmandBfillsBfillingBfilledBfightsBfightingBfightBfiftysBfiftyBfiftiesB	fictionbrBfictionBfetusBferdinyoullBferdinamyhaveBfemmeBfellBfeistyBfeedBfeebleBfedBfeaturesBfeatherBfe5rdinBfayB	favoritesB	favoriteiBfaultBfathersBfataleBfatBfastbrB	fashionedBfarmersBfarmB
farfetchedBfarawayB	fantasiesBfameB	falsenessBfalseBfallsB
faithfullyBfaithBfairBfailsbrBfailsBfailBfactsB	faceliftsB
eyeopeningBeyebrowsBexwifeBexternalizeB
expressionB	expressesB	expressedBexportBexploresBexploredBexploreBexplorationBexploitativeBexploitationBexplodesBexplodeB
explicitlyBexplanationB	explainedBexplainBexpertBexperimentalB
experimentBexperiencesBexpensesBexpenseBexitBexistsBexistingB	exhaustedB	executiveBexecutesBexecsBexdrugBexconsBexconBexcitingB
exceptionsB	exceptionBexcelsBexcellentlyBexceededBexamBexaggeratedBevolveBevokedB	evidencedB	everybodyBeverybitBevermoreparanoidBevansBevanBevaB	europoeanBeurasianBethicalBeternityB
estimationBestateB	essentialBessenceB	espionageBescapesBescapeBeruptsBernestBermBerikaBerabrBequipmentbrBequalBepicsBepicBenvironmentalBentirelyB
enthralledB
entertainsBentertainedB	entertainBensembleB	enigmaticBenhancedBenglandBenemiesBendediB	enchantedBenactsBenableBemptyB	employeesBemployeeBempireBemperorBemotionsBemotionalsexualB	emergenceBembracesBembarrassmentBelmerBelliottBeliteBelicitedB	elegantlyBeleganceBelectricBeldersBelderlyBelderBelamBeileenBeighthBeffortsB	effectsbrBeffectivelyB	effectiveBedwardBeditionBeditedBedgyBedgeBeddieBecstaticBeconomyB	eccentricB	ebullientB	eastwoodsBearnestBearnBeagerlyBdynamismBdwBdvdsBdvBdutchersBduoBdummyBdumberBdukasBducklikeBdubBdrugsBdroppedBdriverBdrippingBdrinkingBdrifterBdressingBdreambrBdreadedBdrawnBdramaticBdrainingBdrainedBdragB	downrightB
downhillbrBdovesBdousesBdoubleentendresBdopeBdoorsBdonutsBdonleveyBdonebrBdonB	dominiqueBdominicBdollopBdollarBdodgyBdodgeBdocumentarysBdocumentarybrBdocumentariesBdoctorsBdoctorB	divulgingBdiversitybrBdistrictBdistressingBdistressBdistractionB
distortionBdispatchBdisorderBdisobedienceB	disjoinedBdisinterestedB
disgustingB	disgustedBdisgracefulBdisconcertingB
disciplineBdiscernBdiscB
disappointBdisabledBdirkBdirectsBdirectBdireB	dinosaursB	dinginessB
diminishesB	dimensionBdiggsBdigBdifferentlyBdifferencesBdifferencebrBdidorBdickBdianneBdianeBdevoursBdevoidBdevilBdeviceB	developedBdevelopBdevastationB	detractedBdetractBdetestsBdetestedB
detestableBdetestB
detectivesBdetailedB	destroyedBdesperatelyBdespairBdesolateBdesiresBdesignedBdeservesBdesecrationBdescentB
descendingBderivesB	depressesB	dependentB
departmentBdentonBdenimBdenatsBdenatB
demonizingBdemonBdemillesBdementedBdeliveryBdeliverBdelinquencyB
delightfulB	deliciousB
deliberateBdegeneratesBdeftlyB
definitionBdefeatBdecrepitBdeclineB
decisionbrBdecencyBdecadesmiloBdecadesBdecadeBdebatingBdeathsBdealsBdealingBdealbrBdeafmuteBdaytimeBdaylightBdawsonsBdaughterBdatingB
datesdeathBdateBdarkerBdaringBdannymichaelBdaniellaBdangerBdammesBdamagedBdamageBdalyBdallasBdaliBdahlBdBczernyBcynicalBcuttingBcutiebrBcutenotBcutelluntilB
customisedBcustomerBcustodyBcurtBcursedBcurrentBcuriositywantingBcueBcubesBcubanBcrushedBcrusadeBcrumbB
cruelfacedBcrucialBcrowdB	crossedbrBcronenbergsBcriticalBcritB	crissakesBcrisisBcringeworthyBcringedBcrilesBcrewsBcreepyBcreditsB
credentialBcreatorB
creativityBcreationBcreatingBcreatesBcrawlbrBcrashingBcrappyBcrankyBcrahanBcplBcozBcovertlyBcovertBcoverBcourageBcouplebrBcounterpartBcostumesBcostumeBcosmosbrBcosB
corruptionB	correctbrBcorrectBcorpsebrBcorporatizationBcoreBcoppolaBcopactiondramaBcoolestB
convincingBconveyedB	convertedBconversationBconventionalB
convenientBcontrivanceBcontributedBcontrastBcontradictoryB
continuityBcontestBcontemporaryBcontainBconstructiveBconstructionB	constructBconstitutionBconstituencyB
constantlyBconstantineBconstantB
conspiringBconsistsBconsistencyB
consideredBconsiderBconsequenceB
conscienceBconradBconnorsBconnoisseursBcongressmanBcongressionalBcongressBconfinedB	confessesB	conductorB
conductingB	conditionB	concludesB
concerningBconceptBconcentratesBconcentrateBconcedesB
concealingBcomputerBcompromisebrB
comprisingBcompositionBcomposerBcomponentbrB
complimentBcomplicationsBcomplicatedB
competitorB	compelledBcompassionateBcompassBcomparedBcompanyB	companiesB	communismBcommunicatorBcommunicationsBcommunicationBcommunicateB	committedB
commitmentBcommissionerB
commercialBcommentatorBcommentBcommendableB	commanderBcomingBcomedicchangeBcombinedBcolumbiaBcolourBcolorbrBcolorBcolonelBcollegefratB
collectionBcollectB	colleagueBcollarBcollaboratingB	coldsweatBcohostsBcoherentBcoffinsBcoercesBcocteausBcocaineBcobraBcoaxingBcmonBclueBclosingBclosedBclooneyBclipBclingingBclimaxBclimaticB	climacticBcliffsBclichésBclichélandmineBclichéBclichesBclicheBcleartoBclaustrophobicB	claudetteBclassyB	classicalBclanB	civilizedB	civiliansBcitizenBcircumstancesB	circuitryBcircleB	cinematicBcinemasB	cinecitt?B
cinderellaBciaBchurchsBchuckleBchuckB
chroniclesBchrisBchoppedBchompBchoiceB
chillingasB	childlikeBchildishB	childhoodBchildenBchildbrBchewBcherryBcheeringBcheckoutBcheckingBcheapenBchasingBchasesuspenseBchasesBcharmedB	charletonBcharactersitemsvehiclesBcharactersbrBcharacterisationsBcharacterdrivenBchaplinsBchannelsBchangingBchangeofpaceBchambermaidsB
challengesB
chalkwhiteB
ceremonialBcenturysBcenturyBcentralBcentersB	celluloidBcellsB	celebrityBcedricBcecilBceaselesslyBcbcBcaveBcatwomanBcatholicBcateringBcategoryBcatchingBcatch22BcatandmouseBcastsBcastleBcashingBcashiersBcashierBcashBcasesbrBcarverBcarryingBcarrierBcarolinaBcarlB
caricatureBcarellBcarefulBcareB	cardboardBcardBcapturesBcapturedBcaptionBcaptainBcapraBcaperBcantonB	canonicalBcanadianBcampyBcampfireB
cameraworkB	camcorderBcaliberB	calculateBcajoleBcageBcableBcBbwBbuttonBbustedBbusinessasusualBbushesBburstingBburnedBbumblingBbullyingBbuiltBbuildBbuffaloBbuffBbuckBbruteBbrushesBbrushBbruceBbrownieBbrownBbrowfurrowingbrBbrowBbrothB
brokendownB	broadcastB	broadbentBbritishBbringingBbrimmingBbrimfullBbrilliantlyB	brightestBbrightBbridgingBbricksBbrennerBbreedBbreathtakingBbreastBbreakBbravoBbraveBbratsBbrandoBbrainsBbrainbrBboyzBboyleB	boyfriendB
boyesquireBboyersBbowsBbowdlerizationBbowdlerBbouncingBboudoirBbottleBbotsBbossellBbossBboroughBborgnineBborelBboredomBbordersBborderBbootyBboomB
booksgamesBbooksBbonusBbonesBbombersBbombBbogeymenBbobBboardsB	boardroomBbmoviesB
blunderingBbluffedB
bluebeardsBblowsBblownBblowingBbloomB	bloodbathBblockBblitzBblithelyBblistB
blindinglyBblindedBblewBblendsBbleepBbleedingBbleakBblatantBblandB	blamelessBblamebrBblairBbladeBbitterB	bislerianBbirdBbiographBbioBbinBbillionB	billboardBbikiniBbikeBbigtimeBbigotedBbignameBbiblicalBbevyBbettyBbetrayalBbetBbestbrBberserkB	bernsteinBberlinBbergmanBbeningB
benefitingBbenefitB
beneficialBbeneathBbenBbeltwayBbelliesBbelievableandBbeingsBbeholdB	behindtheBbehaviorBbeguiledBbeginbrBbeggingBbeexceptBbeenbrB	bedroombrBbedroomBbeddingBbeautiesBbeatyBbeastBbeachesBbbcBbaywatchBbauerBbatBbasisB	basicallyBbaseBbarneyB	barbarityB	banknotesBbankerBbaffledB	backstoryBbabeBawkwardBawfuliBawareBawakesBavoidingBavoidedBavengeBautomobilesB	authorityBauthoritativelyB
australianB	australiaB	aurakotosBauntB	aummmblueB
audiencebrBattractionbrBattitudeBattestBattacksB
attachmentBatrocityB
atrocitiesBatlantaBatkinsBatkindBathletesB
asymmetricB
astoundingBassuresBassuredBassumingBassumedB
assemblingBassBaspectsBaspectBasleepBaskingBaskBasiansBaschenbachsB
aschenbachBartisticBarticleBartfulBartformBartfilmBarrivesB
arrestedbrBarrayB	arousedbrBaroundbrBarmyBarmiesBarmedBargumentBarguablyBareasBareaB
architectsB
archetypesB
archetypalBarcherBappropriatelyBapproachingBapproachBapplicationB
applicableBappearbrB	appealingB	appallingB	apologizeBaotsBanywhereBanywaybrBanyonebrBanymoreBanybodyBantitankB	antihumanBantiaircraftB	anthologyBantagonisticBansweredBannoysB
annoyingbrBanniversaryBannikBannetteBanimatorB	animalsbrBangstBangryBanglesBangeringBangelsbrBangelBancestorBamusingBamusedbrBamountBamileeBamidstBamericasBamericaBambianceBamazedBamandabrBaltersBalteredBalsowhenBalrightBalotBalltimeBallstarBallowsBallowedBallotBallianceBallenBalienriddenB
alienationBalgiersBalfredB	alexandraBalexBalbumBalbinosBalbertBalbeitBalarmingBaircraftBaintBaidesBaidanBahemBaheadBahaBaguayostewartB	agonizingBagobrBagingBagencysBage47BagainbrB
afterwordsB
afterwardsBafootBafghaniB	affectionBadvancedBadvanceBadultBadsBadorableBadditionBaddingB	addictionBaddictBaddBadamB	actressesBactressBactivityBactionfightingBactionbrBactbrBacidBachieveBaccountsBaccomplishesBaccomplishedbrBaccomplishedB
accomplishBaccompanimentB	acclaimedBacclaimB
accidentalBaccidentBacceptsB	acceptingB
acceptanceB
accentuateBaccentsBabsurdBabrahamBabortsBaboardBableBabettedBabeddedB	abandonedB9thB95B9B8510B8B70B60sB60B5thB55B50sB47B40B37B300000B2ndB2discB26B25B24B227thB20sB2006B2003B2000sB1999B1992B1983B1980sB1980B1972B1971B1970sB1968B1956B1955B1947B1939brB1937sB1936B1930sB	18yearoldB16hourB150B14B13thB1229B11B10brB103B10000B100B0510B0
??
Const_5Const*
_output_shapes	
:?)*
dtype0	*??
value??B??	?)"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_26654
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_26659
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?<
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures*
* 

	keras_api* 
;
	keras_api
_lookup_layer
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias*
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
$9 _saved_model_loader_tracked_dict* 
'
1
)2
*3
14
25*
'
0
)1
*2
13
24*
* 
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
* 
?
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratem?)m?*m?1m?2m?v?)v?*v?1v?2v?*

Lserving_default* 
* 
* 
7
M	keras_api
Nlookup_table
Otoken_counts*

Ptrace_0* 

0*

0*
* 
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

]trace_0* 

^trace_0* 

)0
*1*

)0
*1*
* 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
t
v	_imported
w_structured_inputs
x_structured_outputs
y_output_to_inputs_map
z_wrapped_function* 
* 
<
0
1
2
3
4
5
6
7*

{0
|1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
S
}_initializer
~_create_resource
_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?created_variables
?	resources
?trackable_objects
?initializers
?assets
?
signatures
$?_self_saveable_object_factories
ztransform_fn* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 

?serving_default* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
s
serving_default_examplesPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_examples
hash_tableConstConst_1Const_2embedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_25727
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst_6*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_26768
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/embedding/embeddings/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/embedding/embeddings/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_26853Š
?
?
__inference_<lambda>_266548
4key_value_init17612_lookuptableimportv2_table_handle0
,key_value_init17612_lookuptableimportv2_keys2
.key_value_init17612_lookuptableimportv2_values	
identity??'key_value_init17612/LookupTableImportV2?
'key_value_init17612/LookupTableImportV2LookupTableImportV24key_value_init17612_lookuptableimportv2_table_handle,key_value_init17612_lookuptableimportv2_keys.key_value_init17612_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init17612/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?):?)2R
'key_value_init17612/LookupTableImportV2'key_value_init17612/LookupTableImportV2:!

_output_shapes	
:?):!

_output_shapes	
:?)
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_26540

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
o
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25869

review
identity;
ShapeShapereview*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapereview*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:??????????
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:?????????*
dtype0	*
shape:??????????
PartitionedCallPartitionedCallreviewPlaceholderWithDefault:output:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:?????????:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_pruned_25576`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_namereview
?
?
'__inference_dense_3_layer_call_fn_26549

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_25967o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25840

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:??????????
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:?????????*
dtype0	*
shape:??????????
PartitionedCallPartitionedCallinputsPlaceholderWithDefault:output:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:?????????:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_pruned_25576`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
#__inference_signature_wrapper_25584

inputs
inputs_1	
identity

identity_1	?
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2	*:
_output_shapes(
&:?????????:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_pruned_25576`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????b

Identity_1IdentityPartitionedCall:output:1*
T0	*'
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs_1
?
[
8__inference_transform_features_layer_layer_call_fn_26565
inputs_review
identity?
PartitionedCallPartitionedCallinputs_review*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25840`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/review
?
?
__inference_restore_fn_26646
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
v
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_26586
inputs_review
identityB
ShapeShapeinputs_review*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskD
Shape_1Shapeinputs_review*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:??????????
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:?????????*
dtype0	*
shape:??????????
PartitionedCallPartitionedCallinputs_reviewPlaceholderWithDefault:output:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:?????????:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_pruned_25576`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:V R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/review
?
F
__inference__creator_26609
identity: ??MutableHashTable~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_377*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?;
?
__inference__traced_save_26768
file_prefix3
/savev2_embedding_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *)
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?N: : : :: : : : : ::: : : : :	?N: : : ::	?N: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?N:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?i
?
B__inference_model_1_layer_call_and_return_conditional_losses_26293
	review_xfU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_26278:	?N
dense_2_26282: 
dense_2_26284: 
dense_3_26287: 
dense_3_26289:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
tf.reshape_1/ReshapeReshape	review_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_26278*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_25934?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_26282dense_2_26284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_25950?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_26287dense_3_26289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_25967w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	review_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_26509

inputs	)
embedding_lookup_26503:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_26503inputs*
Tindices0	*)
_class
loc:@embedding_lookup/26503*+
_output_shapes
:?????????d*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/26503*+
_output_shapes
:?????????d?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
*
__inference_<lambda>_26659
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?q
?
B__inference_model_1_layer_call_and_return_conditional_losses_26493

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_26471:	?N8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding/embedding_lookup?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_26471?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/26471*+
_output_shapes
:?????????d*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/26471*+
_output_shapes
:?????????d?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????ds
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?q
?
B__inference_model_1_layer_call_and_return_conditional_losses_26419

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_26397:	?N8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding/embedding_lookup?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_26397?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/26397*+
_output_shapes
:?????????d*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/26397*+
_output_shapes
:?????????d?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????ds
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_26560

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_25934

inputs	)
embedding_lookup_25928:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_25928inputs*
Tindices0	*)
_class
loc:@embedding_lookup/25928*+
_output_shapes
:?????????d*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/25928*+
_output_shapes
:?????????d?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?g
?
!__inference__traced_restore_26853
file_prefix8
%assignvariableop_embedding_embeddings:	?N3
!assignvariableop_1_dense_2_kernel: -
assignvariableop_2_dense_2_bias: 3
!assignvariableop_3_dense_3_kernel: -
assignvariableop_4_dense_3_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: B
/assignvariableop_14_adam_embedding_embeddings_m:	?N;
)assignvariableop_15_adam_dense_2_kernel_m: 5
'assignvariableop_16_adam_dense_2_bias_m: ;
)assignvariableop_17_adam_dense_3_kernel_m: 5
'assignvariableop_18_adam_dense_3_bias_m:B
/assignvariableop_19_adam_embedding_embeddings_v:	?N;
)assignvariableop_20_adam_dense_2_kernel_v: 5
'assignvariableop_21_adam_dense_2_bias_v: ;
)assignvariableop_22_adam_dense_3_kernel_v: 5
'assignvariableop_23_adam_dense_3_bias_v:
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_embedding_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_3_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_3_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_2_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_2_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_3_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_3_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
V
:__inference_global_average_pooling1d_1_layer_call_fn_26514

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_embedding_layer_call_fn_26500

inputs	
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_25934s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_26529

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_25950o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_26159
	review_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	review_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_26115o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	review_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?B
?
__inference_adapt_step_19451
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
'__inference_model_1_layer_call_fn_25995
	review_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	review_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_25974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	review_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_transform_features_layer_layer_call_fn_25843

review
identity?
PartitionedCallPartitionedCallreview*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25840`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_namereview
?
?
__inference__initializer_265998
4key_value_init17612_lookuptableimportv2_table_handle0
,key_value_init17612_lookuptableimportv2_keys2
.key_value_init17612_lookuptableimportv2_values	
identity??'key_value_init17612/LookupTableImportV2?
'key_value_init17612/LookupTableImportV2LookupTableImportV24key_value_init17612_lookuptableimportv2_table_handle,key_value_init17612_lookuptableimportv2_keys.key_value_init17612_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init17612/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?):?)2R
'key_value_init17612/LookupTableImportV2'key_value_init17612/LookupTableImportV2:!

_output_shapes	
:?):!

_output_shapes	
:?)
?
?
__inference_save_fn_26638
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
??
?
&__inference_serve_tf_examples_fn_25702
examples]
Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle^
Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
0model_1_text_vectorization_string_lookup_equal_y7
3model_1_text_vectorization_string_lookup_selectv2_t	;
(model_1_embedding_embedding_lookup_25680:	?N@
.model_1_dense_2_matmul_readvariableop_resource: =
/model_1_dense_2_biasadd_readvariableop_resource: @
.model_1_dense_3_matmul_readvariableop_resource: =
/model_1_dense_3_biasadd_readvariableop_resource:
identity??&model_1/dense_2/BiasAdd/ReadVariableOp?%model_1/dense_2/MatMul/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?"model_1/embedding/embedding_lookup?Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB u
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBreviewj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB ?
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0*
Tdense
2*'
_output_shapes
:?????????*
dense_shapes
:*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 x
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:??????????
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:?????????*
dtype0	*
shape:??????????
(transform_features_layer/PartitionedCallPartitionedCall*ParseExample/ParseExampleV2:dense_values:08transform_features_layer/PlaceholderWithDefault:output:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:?????????:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_pruned_25576u
"model_1/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
model_1/tf.reshape_1/ReshapeReshape1transform_features_layer/PartitionedCall:output:0+model_1/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
&model_1/text_vectorization/StringLowerStringLower%model_1/tf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
-model_1/text_vectorization/StaticRegexReplaceStaticRegexReplace/model_1/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite m
,model_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
4model_1/text_vectorization/StringSplit/StringSplitV2StringSplitV26model_1/text_vectorization/StaticRegexReplace:output:05model_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
:model_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
<model_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
<model_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
4model_1/text_vectorization/StringSplit/strided_sliceStridedSlice>model_1/text_vectorization/StringSplit/StringSplitV2:indices:0Cmodel_1/text_vectorization/StringSplit/strided_slice/stack:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
<model_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_1/text_vectorization/StringSplit/strided_slice_1StridedSlice<model_1/text_vectorization/StringSplit/StringSplitV2:shape:0Emodel_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
]model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast=model_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast?model_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdpmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
kmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateromodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0tmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastmmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2nmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuljmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumqmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
hmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2qmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle=model_1/text_vectorization/StringSplit/StringSplitV2:values:0Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
.model_1/text_vectorization/string_lookup/EqualEqual=model_1/text_vectorization/StringSplit/StringSplitV2:values:00model_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
1model_1/text_vectorization/string_lookup/SelectV2SelectV22model_1/text_vectorization/string_lookup/Equal:z:03model_1_text_vectorization_string_lookup_selectv2_tUmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
1model_1/text_vectorization/string_lookup/IdentityIdentity:model_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????y
7model_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
/model_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
>model_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor8model_1/text_vectorization/RaggedToTensor/Const:output:0:model_1/text_vectorization/string_lookup/Identity:output:0@model_1/text_vectorization/RaggedToTensor/default_value:output:0?model_1/text_vectorization/StringSplit/strided_slice_1:output:0=model_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"model_1/embedding/embedding_lookupResourceGather(model_1_embedding_embedding_lookup_25680Gmodel_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_1/embedding/embedding_lookup/25680*+
_output_shapes
:?????????d*
dtype0?
+model_1/embedding/embedding_lookup/IdentityIdentity+model_1/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_1/embedding/embedding_lookup/25680*+
_output_shapes
:?????????d?
-model_1/embedding/embedding_lookup/Identity_1Identity4model_1/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d{
9model_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_1/global_average_pooling1d_1/MeanMean6model_1/embedding/embedding_lookup/Identity_1:output:0Bmodel_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model_1/dense_2/MatMulMatMul0model_1/global_average_pooling1d_1/Mean:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? p
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model_1/dense_3/MatMulMatMul"model_1/dense_2/Relu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_1/dense_3/SigmoidSigmoid model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitymodel_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp#^model_1/embedding/embedding_lookupM^model_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2H
"model_1/embedding/embedding_lookup"model_1/embedding/embedding_lookup2?
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_26619
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_25967

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
,
__inference__destroyer_26604
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
'__inference_model_1_layer_call_fn_26345

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_26115o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_25950

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
:
__inference__creator_26591
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name17613*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?i
?
B__inference_model_1_layer_call_and_return_conditional_losses_25974

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_25935:	?N
dense_2_25951: 
dense_2_25953: 
dense_3_25968: 
dense_3_25970:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_25935*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_25934?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_25951dense_2_25953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_25950?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_25968dense_3_25970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_25967w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
'__inference_model_1_layer_call_fn_26322

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_25974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
R
__inference_pruned_25576

inputs
inputs_1	
identity

identity_1	Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:?????????\
IdentityIdentityinputs_copy:output:0*
T0*'
_output_shapes
:?????????U
inputs_1_copyIdentityinputs_1*
T0	*'
_output_shapes
:?????????`

Identity_1Identityinputs_1_copy:output:0*
T0	*'
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:- )
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????
?	
?
#__inference_signature_wrapper_25727
examples
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_serve_tf_examples_fn_25702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?i
?
B__inference_model_1_layer_call_and_return_conditional_losses_26115

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_26100:	?N
dense_2_26104: 
dense_2_26106: 
dense_3_26109: 
dense_3_26111:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_26100*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_25934?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_26104dense_2_26106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_25950?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_26109dense_3_26111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_25967w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
.
__inference__initializer_26614
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?{
?
 __inference__wrapped_model_25802
	review_xf]
Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle^
Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
0model_1_text_vectorization_string_lookup_equal_y7
3model_1_text_vectorization_string_lookup_selectv2_t	;
(model_1_embedding_embedding_lookup_25780:	?N@
.model_1_dense_2_matmul_readvariableop_resource: =
/model_1_dense_2_biasadd_readvariableop_resource: @
.model_1_dense_3_matmul_readvariableop_resource: =
/model_1_dense_3_biasadd_readvariableop_resource:
identity??&model_1/dense_2/BiasAdd/ReadVariableOp?%model_1/dense_2/MatMul/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?"model_1/embedding/embedding_lookup?Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2u
"model_1/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
model_1/tf.reshape_1/ReshapeReshape	review_xf+model_1/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
&model_1/text_vectorization/StringLowerStringLower%model_1/tf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
-model_1/text_vectorization/StaticRegexReplaceStaticRegexReplace/model_1/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite m
,model_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
4model_1/text_vectorization/StringSplit/StringSplitV2StringSplitV26model_1/text_vectorization/StaticRegexReplace:output:05model_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
:model_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
<model_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
<model_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
4model_1/text_vectorization/StringSplit/strided_sliceStridedSlice>model_1/text_vectorization/StringSplit/StringSplitV2:indices:0Cmodel_1/text_vectorization/StringSplit/strided_slice/stack:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
<model_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_1/text_vectorization/StringSplit/strided_slice_1StridedSlice<model_1/text_vectorization/StringSplit/StringSplitV2:shape:0Emodel_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
]model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast=model_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast?model_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdpmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
kmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateromodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0tmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastmmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2nmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuljmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumqmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
hmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2qmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle=model_1/text_vectorization/StringSplit/StringSplitV2:values:0Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
.model_1/text_vectorization/string_lookup/EqualEqual=model_1/text_vectorization/StringSplit/StringSplitV2:values:00model_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
1model_1/text_vectorization/string_lookup/SelectV2SelectV22model_1/text_vectorization/string_lookup/Equal:z:03model_1_text_vectorization_string_lookup_selectv2_tUmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
1model_1/text_vectorization/string_lookup/IdentityIdentity:model_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????y
7model_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
/model_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
>model_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor8model_1/text_vectorization/RaggedToTensor/Const:output:0:model_1/text_vectorization/string_lookup/Identity:output:0@model_1/text_vectorization/RaggedToTensor/default_value:output:0?model_1/text_vectorization/StringSplit/strided_slice_1:output:0=model_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"model_1/embedding/embedding_lookupResourceGather(model_1_embedding_embedding_lookup_25780Gmodel_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_1/embedding/embedding_lookup/25780*+
_output_shapes
:?????????d*
dtype0?
+model_1/embedding/embedding_lookup/IdentityIdentity+model_1/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_1/embedding/embedding_lookup/25780*+
_output_shapes
:?????????d?
-model_1/embedding/embedding_lookup/Identity_1Identity4model_1/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d{
9model_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_1/global_average_pooling1d_1/MeanMean6model_1/embedding/embedding_lookup/Identity_1:output:0Bmodel_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model_1/dense_2/MatMulMatMul0model_1/global_average_pooling1d_1/Mean:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? p
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model_1/dense_3/MatMulMatMul"model_1/dense_2/Relu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_1/dense_3/SigmoidSigmoid model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitymodel_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp#^model_1/embedding/embedding_lookupM^model_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2H
"model_1/embedding/embedding_lookup"model_1/embedding/embedding_lookup2?
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	review_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?i
?
B__inference_model_1_layer_call_and_return_conditional_losses_26226
	review_xfU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_26211:	?N
dense_2_26215: 
dense_2_26217: 
dense_3_26220: 
dense_3_26222:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
tf.reshape_1/ReshapeReshape	review_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????d       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_26211*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_25934?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_26215dense_2_26217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_25950?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_26220dense_3_26222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_25967w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	review_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_26520

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_25812

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
examples-
serving_default_examples:0?????????>
output_02
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
$9 _saved_model_loader_tracked_dict"
_tf_keras_model
C
1
)2
*3
14
25"
trackable_list_wrapper
C
0
)1
*2
13
24"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
@trace_1
Atrace_2
Btrace_32?
'__inference_model_1_layer_call_fn_25995
'__inference_model_1_layer_call_fn_26322
'__inference_model_1_layer_call_fn_26345
'__inference_model_1_layer_call_fn_26159?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z@trace_1zAtrace_2zBtrace_3
?
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32?
B__inference_model_1_layer_call_and_return_conditional_losses_26419
B__inference_model_1_layer_call_and_return_conditional_losses_26493
B__inference_model_1_layer_call_and_return_conditional_losses_26226
B__inference_model_1_layer_call_and_return_conditional_losses_26293?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
?B?
 __inference__wrapped_model_25802	review_xf"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratem?)m?*m?1m?2m?v?)v?*v?1v?2v?"
	optimizer
,
Lserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
L
M	keras_api
Nlookup_table
Otoken_counts"
_tf_keras_layer
?
Ptrace_02?
__inference_adapt_step_19451?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zPtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Vtrace_02?
)__inference_embedding_layer_call_fn_26500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zVtrace_0
?
Wtrace_02?
D__inference_embedding_layer_call_and_return_conditional_losses_26509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zWtrace_0
':%	?N2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?
]trace_02?
:__inference_global_average_pooling1d_1_layer_call_fn_26514?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z]trace_0
?
^trace_02?
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_26520?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z^trace_0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
?
dtrace_02?
'__inference_dense_2_layer_call_fn_26529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zdtrace_0
?
etrace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_26540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zetrace_0
 : 2dense_2/kernel
: 2dense_2/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?
ktrace_02?
'__inference_dense_3_layer_call_fn_26549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zktrace_0
?
ltrace_02?
B__inference_dense_3_layer_call_and_return_conditional_losses_26560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zltrace_0
 : 2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?
rtrace_0
strace_12?
8__inference_transform_features_layer_layer_call_fn_25843
8__inference_transform_features_layer_layer_call_fn_26565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zrtrace_0zstrace_1
?
ttrace_0
utrace_12?
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_26586
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zttrace_0zutrace_1
?
v	_imported
w_structured_inputs
x_structured_outputs
y_output_to_inputs_map
z_wrapped_function"
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_model_1_layer_call_fn_25995	review_xf"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_26322inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_26345inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_26159	review_xf"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_26419inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_26493inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_26226	review_xf"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_26293	review_xf"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_25727examples"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
g
}_initializer
~_create_resource
_initialize
?_destroy_resourceR jtf.StaticHashTable
O
?_create_resource
?_initialize
?_destroy_resourceR Z

 ??
?B?
__inference_adapt_step_19451iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_embedding_layer_call_fn_26500inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_embedding_layer_call_and_return_conditional_losses_26509inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
:__inference_global_average_pooling1d_1_layer_call_fn_26514inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_26520inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_2_layer_call_fn_26529inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_26540inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_3_layer_call_fn_26549inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_3_layer_call_and_return_conditional_losses_26560inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_transform_features_layer_layer_call_fn_25843review"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_transform_features_layer_layer_call_fn_26565inputs/review"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_26586inputs/review"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25869review"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?created_variables
?	resources
?trackable_objects
?initializers
?assets
?
signatures
$?_self_saveable_object_factories
ztransform_fn"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0B.
__inference_pruned_25576inputsinputs_1
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_26591?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_26599?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_26604?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_26609?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_26614?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_26619?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_26591"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_26599"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_26604"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_26609"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_26614"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_26619"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_25584inputsinputs_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,:*	?N2Adam/embedding/embeddings/m
%:# 2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
%:# 2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
,:*	?N2Adam/embedding/embeddings/v
%:# 2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
%:# 2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
?B?
__inference_save_fn_26638checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_26646restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant6
__inference__creator_26591?

? 
? "? 6
__inference__creator_26609?

? 
? "? 8
__inference__destroyer_26604?

? 
? "? 8
__inference__destroyer_26619?

? 
? "? A
__inference__initializer_26599N???

? 
? "? :
__inference__initializer_26614?

? 
? "? ?
 __inference__wrapped_model_25802uN???)*122?/
(?%
#? 
	review_xf?????????
? "1?.
,
dense_3!?
dense_3?????????j
__inference_adapt_step_19451JO???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
B__inference_dense_2_layer_call_and_return_conditional_losses_26540\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? z
'__inference_dense_2_layer_call_fn_26529O)*/?,
%?"
 ?
inputs?????????
? "?????????? ?
B__inference_dense_3_layer_call_and_return_conditional_losses_26560\12/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_26549O12/?,
%?"
 ?
inputs????????? 
? "???????????
D__inference_embedding_layer_call_and_return_conditional_losses_26509_/?,
%?"
 ?
inputs?????????d	
? ")?&
?
0?????????d
? 
)__inference_embedding_layer_call_fn_26500R/?,
%?"
 ?
inputs?????????d	
? "??????????d?
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_26520{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
:__inference_global_average_pooling1d_1_layer_call_fn_26514nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_26226qN???)*12:?7
0?-
#? 
	review_xf?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_26293qN???)*12:?7
0?-
#? 
	review_xf?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_26419nN???)*127?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_26493nN???)*127?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_1_layer_call_fn_25995dN???)*12:?7
0?-
#? 
	review_xf?????????
p 

 
? "???????????
'__inference_model_1_layer_call_fn_26159dN???)*12:?7
0?-
#? 
	review_xf?????????
p

 
? "???????????
'__inference_model_1_layer_call_fn_26322aN???)*127?4
-?*
 ?
inputs?????????
p 

 
? "???????????
'__inference_model_1_layer_call_fn_26345aN???)*127?4
-?*
 ?
inputs?????????
p

 
? "???????????
__inference_pruned_25576?~?{
t?q
o?l
1
review'?$
inputs/review?????????
7
	sentiment*?'
inputs/sentiment?????????	
? "m?j
0
	review_xf#? 
	review_xf?????????
6
sentiment_xf&?#
sentiment_xf?????????	y
__inference_restore_fn_26646YOK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_26638?O&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
#__inference_signature_wrapper_25584?i?f
? 
_?\
*
inputs ?
inputs?????????
.
inputs_1"?
inputs_1?????????	"m?j
0
	review_xf#? 
	review_xf?????????
6
sentiment_xf&?#
sentiment_xf?????????	?
#__inference_signature_wrapper_25727~N???)*129?6
? 
/?,
*
examples?
examples?????????"3?0
.
output_0"?
output_0??????????
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_25869?>?;
4?1
/?,
*
review ?
review?????????
? "A?>
7?4
2
	review_xf%?"
0/review_xf?????????
? ?
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_26586?E?B
;?8
6?3
1
review'?$
inputs/review?????????
? "A?>
7?4
2
	review_xf%?"
0/review_xf?????????
? ?
8__inference_transform_features_layer_layer_call_fn_25843w>?;
4?1
/?,
*
review ?
review?????????
? "5?2
0
	review_xf#? 
	review_xf??????????
8__inference_transform_features_layer_layer_call_fn_26565~E?B
;?8
6?3
1
review'?$
inputs/review?????????
? "5?2
0
	review_xf#? 
	review_xf?????????